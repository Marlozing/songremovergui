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
        icon: path.join(__dirname, 'assets/icon.ico'),
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: true,
            contextIsolation: false
        }
    });
    mainWindow.setMenu(null);
    mainWindow.loadFile('index.html');
}

function runCommand(command, args, statusPrefix) {
    return new Promise((resolve, reject) => {
        const startMsg = `실행: ${command} ${args.join(' ')}`;
        console.log(startMsg);

        const spawnOptions = { windowsHide: true };
        const proc = spawn(command, args, spawnOptions);

        proc.stdout.on('data', (data) => {
            const lines = data.toString().split('\n');
            lines.forEach(line => {
                if (line.trim() !== '') {
                    console.log(`[${statusPrefix}] ${line.trim()}`);
                    if (mainWindow) mainWindow.webContents.send('log-msg', `[${statusPrefix}] ${line.trim()}`);
                }
            });
        });

        proc.stderr.on('data', (data) => {
            const errorMsg = data.toString().trim();
            console.error(`[${statusPrefix} Error] ${errorMsg}`);
            if (mainWindow) mainWindow.webContents.send('log-msg', `[${statusPrefix} ERROR] ${errorMsg}`);
        });

        proc.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`${statusPrefix} failed with code ${code}`));
        });
    });
}

app.whenReady().then(() => {
    createWindow();

    ipcMain.handle('dialog:openFile', async () => {
        const { canceled, filePaths } = await dialog.showOpenDialog({
            properties: ['openFile'],
            filters: [{ name: 'Audio', extensions: ['wav', 'mp3'] }]
        });
        if (canceled) return null;
        return filePaths[0];
    });

    ipcMain.handle('process:audio', async (event, refPath, micPath, alpha, beta, useDeepFilter) => {
        if (alpha <= beta) {
            return { success: false, error: '설정 오류: Alpha > Beta 여야 합니다.' };
        }

        const tempDir = os.tmpdir();
        const jobDirName = `sr_job_${Date.now()}`;
        const jobDirPath = path.join(tempDir, jobDirName);

        if (!fs.existsSync(jobDirPath)) fs.mkdirSync(jobDirPath);

        // 입력 파일명 (임시)
        const tempFileName = "temp_process.wav";
        const tempFilePath = path.join(jobDirPath, tempFileName);

        try {
            let pyCommand, dfBinPath;

            if (app.isPackaged) {
                pyCommand = path.join(process.resourcesPath, 'align_audio.exe');
                dfBinPath = path.join(process.resourcesPath, 'deep-filter.exe');
            } else {
                pyCommand = 'python';
                dfBinPath = path.join(__dirname, 'resources', 'deep-filter.exe');
            }

            // --- Step 1: Python 전처리 ---
            if(mainWindow) mainWindow.webContents.send('log-msg', `>>> 1단계: Python 전처리 시작 (Alpha=${alpha}, Beta=${beta})`);

            let pyArgs;
            if (app.isPackaged) {
                pyArgs = [refPath, micPath, tempFilePath, String(alpha), String(beta)];
            } else {
                pyArgs = ['align_audio.py', refPath, micPath, tempFilePath, String(alpha), String(beta)];
            }

            await runCommand(pyCommand, pyArgs, 'Python');

            // --- Step 2: AI 노이즈 제거 ---
            let finalSourceFile = tempFilePath;

            if (useDeepFilter) {
                if(mainWindow) mainWindow.webContents.send('log-msg', `>>> 2단계: AI 노이즈 제거 (DeepFilterNet) 시작`);

                // DeepFilterNet 실행 (결과는 jobDirPath 내에 생성됨)
                await runCommand(dfBinPath, [tempFilePath, '-o', jobDirPath], 'DeepFilter');

                // [수정] 파일 찾기 로직 강화
                // DeepFilterNet이 생성한 파일은 입력 파일명(temp_process.wav)이 아니면서 .wav로 끝나는 파일임
                const files = fs.readdirSync(jobDirPath);
                if(mainWindow) mainWindow.webContents.send('log-msg', `[Error] 폴더 내 파일 목록: ${files.join(', ')} ${jobDirPath}`);
                const outputFiles = files.filter(f => f.endsWith('.wav'));

                if (outputFiles.length > 0) {
                    // 첫 번째로 발견된 새로운 wav 파일을 결과물로 간주
                    finalSourceFile = path.join(jobDirPath, outputFiles[0]);
                    if(mainWindow) mainWindow.webContents.send('log-msg', `[Info] 생성된 파일 발견: ${outputFiles[0]}`);
                } else {
                    // 파일이 없으면 목록을 로그에 출력해서 디버깅 도움
                    if(mainWindow) mainWindow.webContents.send('log-msg', `[Error] 폴더 내 파일 목록: ${files.join(', ')}`);
                    throw new Error('AI 처리는 완료되었으나 결과 파일(WAV)을 찾을 수 없습니다.');
                }
            } else {
                if(mainWindow) mainWindow.webContents.send('log-msg', `>>> 2단계 건너뜀 (AI 사용 안 함)`);
            }

            // --- 완료 ---
            if (fs.existsSync(finalSourceFile)) {``
                if(mainWindow) mainWindow.webContents.send('log-msg', `>>> [SUCCESS] 처리 완료! 하단 버튼을 눌러 저장하세요.`);
                return { success: true, tempPath: finalSourceFile };
            } else {
                throw new Error('결과 파일을 찾을 수 없습니다.');
            }

        } catch (error) {
            console.error(error);
            if(mainWindow) mainWindow.webContents.send('log-msg', `[CRITICAL ERROR] ${error.message}`);
            try {
                if (fs.existsSync(jobDirPath)) fs.rmSync(jobDirPath, { recursive: true, force: true });
            } catch (cleanupErr) {}
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('save:file', async (event, sourcePath) => {
        const { canceled, filePath: savePath } = await dialog.showSaveDialog({
            title: '결과물 저장',
            defaultPath: 'cleaned_audio.wav',
            filters: [{ name: 'WAV Audio', extensions: ['wav'] }]
        });

        if (canceled || !savePath) return { success: false };

        try {
            fs.copyFileSync(sourcePath, savePath);
            // 저장 후 임시 폴더 삭제
            const jobDirPath = path.dirname(sourcePath);
            fs.rmSync(jobDirPath, { recursive: true, force: true });
            return { success: true, path: savePath };
        } catch (e) {
            return { success: false, error: e.message };
        }
    });
});

app.on('window-all-closed', () => { app.quit(); });