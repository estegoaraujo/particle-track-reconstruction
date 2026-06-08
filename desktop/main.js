
const { app, BrowserWindow } = require('electron')
const path = require('path')

function createWindow() {
    // Create the browser window
    const win = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        title: 'Particle Track Reconstruction',
        backgroundColor: '#0a0a0a',
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    })

    // Load the HTML file
    win.loadFile('src/index.html')

    // Open DevTools in development
    // win.webContents.openDevTools()
}

// Create window when app is ready
app.whenReady().then(() => {
    createWindow()

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

// Quit when all windows are closed
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit()
    }
})
