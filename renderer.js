const { ipcRenderer } = require('electron');

// region DOM 요소 가져오기
const mainView = document.getElementById('main-view');
const logView = document.getElementById('log-view');
const logContainer = document.getElementById('log-container');
const downloadArea = document.getElementById('download-area');
const btnDownload = document.getElementById('btn-download');

const dropRef = document.getElementById('drop-ref');
const nameRef = document.getElementById('name-ref');
const dropMic = document.getElementById('drop-mic');
const nameMic = document.getElementById('name-mic');
const btnProcess = document.getElementById('btn-process');

const settingsPanel = document.getElementById('settingsPanel');
const btnToggleSettings = document.getElementById('btnToggleSettings');
const rangeAlpha = document.getElementById('rangeAlpha');
const numAlpha = document.getElementById('numAlpha');
const rangeBeta = document.getElementById('rangeBeta');
const numBeta = document.getElementById('numBeta');
const checkUseDeepFilter = document.getElementById('checkUseDeepFilter');
// endregion

let refPath = null;
let micPath = null;
let currentTempFile = null; // 완료된 임시 파일 경로 저장용

// region 드래그 앤 드롭 핸들러
function setupDragAndDrop(dropZone, nameElement, isRef) {
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('active'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('active'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('active');
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (isRef) { refPath = file.path; nameElement.innerText = file.name; nameElement.classList.add('filename'); }
            else { micPath = file.path; nameElement.innerText = file.name; nameElement.classList.add('filename'); }
        }
    });
    dropZone.addEventListener('click', async () => {
        const result = await ipcRenderer.invoke('dialog:openFile');
        if (result) {
            if (isRef) { refPath = result; nameRef.innerText = result.split(/[\\/]/).pop(); nameRef.classList.add('filename'); }
            else { micPath = result; nameMic.innerText = result.split(/[\\/]/).pop(); nameMic.classList.add('filename'); }
        }
    });
}
setupDragAndDrop(dropRef, nameRef, true);
setupDragAndDrop(dropMic, nameMic, false);
// endregion

// region 설정 UI (Wall Logic)
btnToggleSettings.addEventListener('click', () => settingsPanel.classList.toggle('open'));

function updateValue(element, value) { element.value = parseFloat(value).toFixed(1); }

function onAlphaChange(e) {
    let aVal = parseFloat(e.target.value);
    let bVal = parseFloat(rangeBeta.value);
    const limit = bVal + 0.1;
    if (aVal <= bVal) aVal = limit;
    updateValue(rangeAlpha, aVal);
    updateValue(numAlpha, aVal);
}

function onBetaChange(e) {
    let bVal = parseFloat(e.target.value);
    let aVal = parseFloat(rangeAlpha.value);
    const limit = aVal - 0.1;
    if (bVal >= aVal) bVal = limit;
    updateValue(rangeBeta, bVal);
    updateValue(numBeta, bVal);
}

rangeAlpha.addEventListener('input', onAlphaChange);
numAlpha.addEventListener('input', onAlphaChange);
rangeBeta.addEventListener('input', onBetaChange);
numBeta.addEventListener('input', onBetaChange);
// endregion

// region [화면 전환 및 로그 출력]
function appendLog(msg) {
    const div = document.createElement('div');
    div.className = 'log-entry';

    if (msg.includes('[ERROR]') || msg.includes('Error:')) div.classList.add('log-error');
    else if (msg.includes('SUCCESS') || msg.includes('완료')) div.classList.add('log-success');
    else if (msg.includes('단계') || msg.includes('>>>')) div.classList.add('log-python');
    else div.classList.add('log-info');

    div.innerText = msg;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
}

ipcRenderer.on('log-msg', (event, msg) => {
    appendLog(msg);
});

// 변환 시작 버튼
btnProcess.addEventListener('click', async () => {
    if (!refPath || !micPath) { alert("오디오 파일 두 개를 모두 지정해주세요."); return; }
    const alpha = parseFloat(numAlpha.value);
    const beta = parseFloat(numBeta.value);
    const useDeepFilter = checkUseDeepFilter.checked;
    if (alpha <= beta) { alert("오류: Alpha는 Beta보다 커야 합니다."); return; }

    // 화면 전환 및 초기화
    mainView.style.display = 'none';
    logView.style.display = 'flex';
    logContainer.innerHTML = '';
    downloadArea.style.display = 'none'; // 다운로드 버튼 숨김
    btnProcess.disabled = true;

    try {
        // [수정] 처리가 끝나면 경로만 받아옴 (저장 안 함)
        const result = await ipcRenderer.invoke('process:audio', refPath, micPath, alpha, beta, useDeepFilter);

        if (result.success) {
            currentTempFile = result.tempPath;
            // 성공 시 다운로드 버튼 표시
            downloadArea.style.display = 'block';
        } else {
            appendLog(`\n[오류] 작업 실패: ${result.error}`);
            setTimeout(() => {
                logView.style.display = 'none';
                mainView.style.display = 'flex';
            }, 3000);
        }
    } catch (error) {
        appendLog(`[CRITICAL] 오류: ${error}`);
        setTimeout(() => {
            logView.style.display = 'none';
            mainView.style.display = 'flex';
        }, 3000);
    } finally {
        btnProcess.disabled = false;
    }
});

// [추가] 파일 내려받기 버튼 핸들러
btnDownload.addEventListener('click', async () => {
    if (!currentTempFile) return;

    const result = await ipcRenderer.invoke('save:file', currentTempFile);

    if (result.success) {
        alert('저장이 완료되었습니다.\n메인 화면으로 돌아갑니다.');
        // 화면 복귀
        logView.style.display = 'none';
        mainView.style.display = 'flex';
        downloadArea.style.display = 'none';
        currentTempFile = null;
    } else if (result.error) {
        alert('저장 실패: ' + result.error);
    }
    // 취소 시에는 아무 일도 안 함 (로그 화면 유지)
});
// endregion