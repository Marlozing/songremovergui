const { ipcRenderer } = require('electron');

// region DOM 요소
const views = {
    main: document.getElementById('main-view'),
    log: document.getElementById('log-view')
};
const zones = {
    ref: document.getElementById('drop-ref'),
    mic: document.getElementById('drop-mic')
};
const names = {
    ref: document.getElementById('name-ref'),
    mic: document.getElementById('name-mic')
};
const btns = {
    process: document.getElementById('btn-process'),
    settings: document.getElementById('btnToggleSettings'),
    download: document.getElementById('btn-download')
};
const settings = {
    panel: document.getElementById('settingsPanel'),
    alpha: document.getElementById('rangeAlpha'),
    alphaNum: document.getElementById('numAlpha'),
    beta: document.getElementById('rangeBeta'),
    betaNum: document.getElementById('numBeta'),
    deepFilter: document.getElementById('checkUseDeepFilter')
};
// endregion

let state = { ref: null, mic: null, temp: null };

// region 드래그 앤 드롭
function setupDrop(zone, nameElem, key) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('active'); });
    zone.addEventListener('dragleave', (e) => { e.preventDefault(); zone.classList.remove('active'); });
    zone.addEventListener('drop', (e) => {
        e.preventDefault(); zone.classList.remove('active');
        if (e.dataTransfer.files.length) {
            const f = e.dataTransfer.files[0];
            state[key] = f.path;
            nameElem.innerText = f.name;
            nameElem.classList.add('filename');
        }
    });
    zone.addEventListener('click', async () => {
        const path = await ipcRenderer.invoke('dialog:openFile');
        if (path) {
            state[key] = path;
            nameElem.innerText = path.split(/[\\/]/).pop();
            nameElem.classList.add('filename');
        }
    });
}
setupDrop(zones.ref, names.ref, 'ref');
setupDrop(zones.mic, names.mic, 'mic');
// endregion

// region 설정 로직
btns.settings.addEventListener('click', (e) => {
    e.stopPropagation();
    settings.panel.classList.toggle('open');
});
document.addEventListener('click', (e) => {
    if (!settings.panel.contains(e.target) && !btns.settings.contains(e.target)) {
        settings.panel.classList.remove('open');
    }
});

// 슬라이더 연동
settings.alpha.addEventListener('input', (e) => {
    let a = parseFloat(e.target.value);
    let b = parseFloat(settings.beta.value);
    if(a <= b) a = b + 0.1;
    e.target.value = a;
    settings.alphaNum.value = a.toFixed(1);
});
settings.beta.addEventListener('input', (e) => {
    let b = parseFloat(e.target.value);
    let a = parseFloat(settings.alpha.value);
    if(b >= a) b = a - 0.1;
    e.target.value = b;
    settings.betaNum.value = b.toFixed(1);
});
// endregion

// region 실행 및 로그
function log(msg) {
    const div = document.createElement('div');
    div.innerText = msg;
    div.style.marginBottom = "5px";
    if(msg.includes('ERR')) div.style.color = "#ff6b6b";
    else if(msg.includes('>>>')) div.style.color = "#4facfe";
    document.getElementById('log-container').appendChild(div);
}
ipcRenderer.on('log-msg', (e, msg) => log(msg));

btns.process.addEventListener('click', async () => {
    if (!state.ref || !state.mic) return alert("파일을 모두 선택해주세요.");

    views.main.style.display = 'none';
    views.log.style.display = 'flex';
    document.getElementById('log-container').innerHTML = '';
    document.getElementById('download-area').style.display = 'none';

    const res = await ipcRenderer.invoke('process:audio',
        state.ref, state.mic,
        parseFloat(settings.alpha.value),
        parseFloat(settings.beta.value),
        settings.deepFilter.checked
    );

    if (res.success) {
        state.temp = res.tempPath;
        log("\n[완료] 변환 성공!");
        document.getElementById('download-area').style.display = 'block';
    } else {
        log(`[ERROR] ${res.error}`);
        setTimeout(() => { views.log.style.display = 'none'; views.main.style.display = 'flex'; }, 3000);
    }
});

btns.download.addEventListener('click', async () => {
    if(!state.temp) return;
    const res = await ipcRenderer.invoke('save:file', state.temp);
    if(res.success) {
        alert("저장되었습니다.");
        views.log.style.display = 'none';
        views.main.style.display = 'flex';
        state.temp = null;
    }
});
// endregion