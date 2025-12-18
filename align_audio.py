import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import os
from numba import jit  # Numba 임포트
import warnings
import sys
import io
import multiprocessing

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
warnings.filterwarnings("ignore")

# region 정규화 및 보조 함수

# 프론트엔드 메세지 전송
def send_status(message):
    print(f"STATUS:{message}", flush=True)

# 크거나 같은 2의 제곱수 계산 함수
def next_pow2(n):
    return 1 << (int(n - 1).bit_length())


# 오디오 정규화 함수
def robust_normalize(data):
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    max_val = np.max(np.abs(data))
    if max_val > 1e-5: data = data / max_val
    return data


# endregion

# region 정렬 관련 함수들

# region gcc-phat 함수

def calculate_gcc_phat(x, y):
    # FFT 크기 계산
    n = len(x) + len(y) - 1
    n_fft = next_pow2(n)

    # FFT 수행
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)

    # PHAT 가중치를 적용한 상호 전력 스펙트럼
    G = X * np.conj(Y)
    R = G / (np.abs(G) + 1e-12)

    # 역 FFT로 상호 상관 함수 계산
    cc = np.fft.irfft(R, n=n_fft)

    # 지연이 0인 지점을 중심으로 재정렬
    half = n_fft // 2
    cc_lin = np.concatenate((cc[-half:], cc[:half + 1]))
    k = int(np.argmax(cc_lin))

    # 서브 샘플 보간 (정밀도 향상)
    delta = 0.0
    if 0 < k < len(cc_lin) - 1:
        y1, y2, y3 = cc_lin[k - 1], cc_lin[k], cc_lin[k + 1]
        d = (y1 - 2 * y2 + y3)
        delta = 0.0 if abs(d) < 1e-20 else 0.5 * (y1 - y3) / d
        delta = float(np.clip(delta, -0.5, 0.5))

    lag_samples = (k + delta) - (len(cc_lin) - 1) / 2.0
    return int(round(lag_samples))


# endregion

# region 정밀 지연 보정 함수

def refine_lag_robust(ref, mic, initial_lag, search_range=200, keep_ratio=0.7):
    n_ref = len(ref)
    center = initial_lag
    lags = range(center - search_range, center + search_range + 1)

    # 효율적인 뷰 생성을 위한 선행 패딩
    pad_size = abs(center) + search_range + 1000
    mic_padded = np.pad(mic, (pad_size, pad_size), 'constant')

    # 속도 최적화를 위해 앞부분 30초만 비교
    compare_len = min(n_ref, 16000 * 30)
    ref_comp = ref[:compare_len]
    k = int(compare_len * keep_ratio)  # 하위 70% 인덱스

    best_lag = center
    min_error = float('inf')

    for lag in lags:
        start = pad_size + lag
        end = start + compare_len
        mic_view = mic_padded[start:end]

        if len(mic_view) < compare_len: continue

        # L1 오차 계산
        diff = np.abs(mic_view - ref_comp)

        # 하위 70% 오차만 합산 (목소리 제외)
        partitioned = np.partition(diff, k)
        err = np.sum(partitioned[:k])

        if err < min_error:
            min_error = err
            best_lag = lag

    send_status(f"지연 시간 계산 완료")
    return best_lag


# endregion

# region 오디오 정렬 함수

def align_ref_to_mic_canvas(ref, mic_len, lag):
    ref_aligned = np.zeros(mic_len, dtype=np.float32)
    n_ref = len(ref)

    start_idx = lag

    # 복사 범위 계산
    r_start = max(0, -start_idx)
    r_end = min(n_ref, mic_len - start_idx)
    m_start = max(0, start_idx)
    m_end = min(mic_len, start_idx + n_ref)

    copy_len = min(r_end - r_start, m_end - m_start)

    if copy_len > 0:
        ref_aligned[m_start: m_start + copy_len] = ref[r_start: r_start + copy_len]

    send_status(f"오디오 정렬 완료")
    return ref_aligned


# endregion

# endregion

# region 제거 관련 함수들

# region 배경음 제거 함수

def wiener_filter_soft(ref, mic, alpha=0.5, beta=0.2):
    # 고해상도 설정
    N_FFT = 4096
    HOP_LENGTH = 512

    f, t, Z_ref = signal.stft(ref, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    _, _, Z_mic = signal.stft(mic, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)

    P_ref = np.abs(Z_ref) ** 2
    P_mic = np.abs(Z_mic) ** 2 + 1e-12

    subtracted_power = P_mic - (alpha * P_ref)

    floor = P_mic * beta
    P_estimated = np.maximum(subtracted_power, floor)

    mask = P_estimated / P_mic
    mask = np.sqrt(mask)

    Z_clean = Z_mic * mask
    _, clean_audio = signal.istft(Z_clean, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)

    return clean_audio


# endregion

# region 후처리 함수

# region 파이썬 -> 기계어 함수

@jit(nopython=True, cache=True)
def _calculate_gain_curve_jit(abs_audio, threshold_linear, ratio, gain_decay, env_decay):
    n_samples = len(abs_audio)
    gain_curve = np.zeros(n_samples, dtype=np.float32)
    current_env = 0.0
    current_gain = 1.0

    for i in range(n_samples):
        # 엔벨로프 추적 (빠른 반응)
        val = abs_audio[i]
        if val > current_env:
            current_env = val
        else:
            current_env = current_env * env_decay + val * (1.0 - env_decay)

        # 목표 게인 설정
        if current_env > threshold_linear:
            target_gain = 1.0
        else:
            target_gain = ratio

        # 게인 적용 (Attack은 즉시, Release는 천천히)
        if target_gain > current_gain:
            current_gain = target_gain  # Attack
        else:
            current_gain = current_gain * gain_decay  # Release
            if current_gain < ratio: current_gain = ratio

        gain_curve[i] = current_gain

    return gain_curve

# endregion

def apply_soft_expander(audio, threshold_db=-45.0, ratio=0.2, release_ms=400, fs=48000):
    threshold_linear = 10 ** (threshold_db / 20)

    # Numba 처리를 위해 float32 타입 보장
    abs_audio = np.abs(audio).astype(np.float32)

    # 감쇠 계수 계산 (문 닫는 속도)
    if release_ms > 0:
        release_samples = int((release_ms / 1000) * fs)
        gain_decay = np.exp(-1.0 / release_samples)
    else:
        gain_decay = 0.0

    # 엔벨로프 추적용 감쇠 계수 (센서 반응 속도 - 10ms 고정)
    env_decay = np.exp(-1.0 / (fs * 0.01))

    # [변경] 분리된 Numba JIT 함수 호출 (속도 가속 구간)
    gain_curve = _calculate_gain_curve_jit(abs_audio, threshold_linear, ratio, gain_decay, env_decay)

    # 팝 노이즈 방지용 추가 스무딩 (Numpy Convolve는 이미 빠르므로 유지)
    kernel_size = 500
    gain_curve_smooth = np.convolve(gain_curve, np.ones(kernel_size) / kernel_size, mode='same')

    return audio * gain_curve_smooth


# endregion

# endregion

# region 메인 함수

def align_audio(ref_path, mic_path, out_path, alpha=None, beta=None):
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {ref_path}")
    if not os.path.exists(mic_path):
        raise FileNotFoundError(f"타겟 파일을 찾을 수 없습니다: {mic_path}")

    fs_ref, data_ref = wav.read(ref_path)
    fs_mic, data_mic = wav.read(mic_path)

    if fs_ref != fs_mic:
        raise ValueError(f"샘플링 레이트가 일치하지 않습니다. (Ref: {fs_ref}, Mic: {fs_mic})")

    fs = fs_mic

    # 스테레오를 모노로 변환 (정렬 계산용)
    if data_ref.ndim > 1:
        data_ref_mono = np.mean(data_ref, axis=1)
    else:
        data_ref_mono = data_ref

    if data_mic.ndim > 1:
        data_mic_mono = np.mean(data_mic, axis=1)
    else:
        data_mic_mono = data_mic

    # 정규화 (정렬 계산용 모노 데이터)
    ref_mono = robust_normalize(data_ref_mono)
    mic_mono = robust_normalize(data_mic_mono)

    # 실제 처리를 위한 원본 정규화 (채널 유지)
    ref_full = robust_normalize(data_ref)
    mic_full = robust_normalize(data_mic)

    # 2. 정렬 (Alignment)
    # GCC-PHAT으로 초기값 탐색 (모노 기준)
    gcc_lag = calculate_gcc_phat(mic_mono, ref_mono)
    # 이상치 제거를 통한 정밀 보정 (모노 기준)
    best_lag = refine_lag_robust(ref_mono, mic_mono, initial_lag=gcc_lag, search_range=200)

    # 3. 분리 및 후처리 (채널별 처리)
    # 입력이 스테레오인 경우 채널별로 분리하여 처리
    if mic_full.ndim == 1:
        # 모노인 경우
        mic_channels = [mic_full]
        ref_channels = [ref_full]
    else:
        # 스테레오인 경우 (채널 분리)
        mic_channels = [mic_full[:, ch] for ch in range(mic_full.shape[1])]
        # Ref가 모노인데 Mic가 스테레오면 Ref를 복제, 둘 다 스테레오면 분리
        if ref_full.ndim == 1:
            ref_channels = [ref_full for _ in range(len(mic_channels))]
        else:
            ref_channels = [ref_full[:, ch] for ch in range(ref_full.shape[1])]

    processed_channels = []

    for i in range(len(mic_channels)):
        mic_ch = mic_channels[i]
        ref_ch = ref_channels[i]

        # 캔버스 정렬 (채널별)
        ref_aligned = align_ref_to_mic_canvas(ref_ch, len(mic_ch), best_lag)

        # 고음질 위너 필터 적용
        cleaned_ch = wiener_filter_soft(ref_aligned, mic_ch, alpha, beta)

        # ISTFT 후 길이 보정
        if len(cleaned_ch) > len(mic_ch):
            cleaned_ch = cleaned_ch[:len(mic_ch)]
        elif len(cleaned_ch) < len(mic_ch):
            cleaned_ch = np.pad(cleaned_ch, (0, len(mic_ch) - len(cleaned_ch)), 'constant')

        # 후처리 (소프트 익스팬더) - Numba 적용으로 가속됨
        final_ch = apply_soft_expander(cleaned_ch, threshold_db=-45.0, ratio=0.2, release_ms=400, fs=fs)

        send_status(f"채널 {i+1} 후처리 완료")
        processed_channels.append(final_ch)

    # 채널 병합
    if len(processed_channels) > 1:
        final_audio = np.stack(processed_channels, axis=1)
    else:
        final_audio = processed_channels[0]

    # 5. 저장
    wav.write(out_path, fs, np.int16(final_audio * 32767))


# endregion

if __name__ == "__main__":
    import sys

    multiprocessing.freeze_support()
    
    # Electron에서 인자 3개를 보낼 것입니다: [스크립트명, Ref경로, Mic경로, 저장경로]
    if len(sys.argv) < 6:
        print("Error: 인자가 부족합니다. (Usage: python align_audio.py ref.wav mic.wav out.wav alpha beta)")
        sys.exit(1)

    ref_path = sys.argv[1]
    mic_path = sys.argv[2]
    out_path = sys.argv[3]

    try:
        alpha = float(sys.argv[4])
        beta = float(sys.argv[5])
    except ValueError:
        print("Error: Alpha와 Beta 값은 숫자여야 합니다.")
        sys.exit(1)

    print(f"STATUS: 처리 시작... (Target: {out_path})")
    
    try:
        align_audio(ref_path, mic_path, out_path, alpha, beta)
        
        # Electron에게 성공 신호를 보냄 (stdout)
        print("SUCCESS: 완료")
    except Exception as e:
        # 에러 내용을 출력하여 Electron이 잡을 수 있게 함
        print(f"ERROR: {str(e)}")
        sys.exit(1)