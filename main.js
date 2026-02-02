const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        title: "Song Remover",
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });
    mainWindow.setMenu(null);
    mainWindow.loadFile('index.html');
}

// 명령어 실행 헬퍼
function runCommand(command, args, prefix) {
    return new Promise((resolve, reject) => {
        const proc = spawn(command, args, {
            windowsHide: true,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
        });

        proc.stdout.on('data', (data) => {
            const line = data.toString().trim();
            if (line && mainWindow) mainWindow.webContents.send('log-msg', `[${prefix}] ${line}`);
        });

        proc.stderr.on('data', (data) => {
            const err = data.toString().trim();
            if (mainWindow) mainWindow.webContents.send('log-msg', `[${prefix} ERR] ${err}`);
        });

        proc.on('close', (code) => code === 0 ? resolve() : reject(new Error(`Code ${code}`)));
    });
}

app.whenReady().then(() => {
    createWindow();

    ipcMain.handle('dialog:openFile', async () => {
        const { canceled, filePaths } = await dialog.showOpenDialog({
            filters: [{ name: 'Audio', extensions: ['wav', 'mp3'] }]
        });
        return canceled ? null : filePaths[0];
    });

    ipcMain.handle('process:audio', async (event, refPath, micPath, alpha, beta, useDeepFilter) => {
        const jobDir = path.join(os.tmpdir(), `sr_${Date.now()}`);
        if (!fs.existsSync(jobDir)) fs.mkdirSync(jobDir);
        const tempWav = path.join(jobDir, "temp_step1.wav");

        try {
            const pyCmd = app.isPackaged ? path.join(process.resourcesPath, 'align_audio.exe') : 'python';
            const dfCmd = app.isPackaged ? path.join(process.resourcesPath, 'deep-filter.exe') : path.join(__dirname, 'resources', 'deep-filter.exe');

            // 1단계: Python 실행
            if(mainWindow) mainWindow.webContents.send('log-msg', '>>> 1단계: 정렬 및 제거 시작');
            const pyArgs = app.isPackaged
                ? [refPath, micPath, tempWav, String(alpha), String(beta)]
                : ['-u', 'align_audio.py', refPath, micPath, tempWav, String(alpha), String(beta)];

            await runCommand(pyCmd, pyArgs, 'Python');

            // 2단계: AI 실행
            let finalFile = tempWav;
            if (useDeepFilter) {
                if(mainWindow) mainWindow.webContents.send('log-msg', '>>> 2단계: AI 노이즈 제거 시작');
                await runCommand(dfCmd, [tempWav, '-o', jobDir], 'DeepFilter');

                // 결과 파일 찾기
                const files = fs.readdirSync(jobDir);
                const out = files.find(f => f.endsWith('.wav') && f !== "temp_step1.wav");
                if (out) finalFile = path.join(jobDir, out);
            }

            return { success: true, tempPath: finalFile };
        } catch (error) {
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('save:file', async (event, sourcePath) => {
        const { canceled, filePath } = await dialog.showSaveDialog({ defaultPath: 'result.wav' });
        if (canceled) return { success: false };

        fs.copyFileSync(sourcePath, filePath);
        fs.rmSync(path.dirname(sourcePath), { recursive: true, force: true });
        return { success: true };
    });
});