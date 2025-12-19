  const { app, BrowserWindow, Menu } = require('electron');

   // Create a menu template
   const menuTemplate = [
     {
       label: 'File',
       submenu: [
         {
           label: 'Open',
           click() {
             console.log('Open clicked!');
           }
         },
         {
           label: 'Save',
           click() {
             console.log('Save clicked!');
           }
         },
         {
           label: 'Quit',
           click() {
             app.quit();
           }
         }
       ]
     },
     {
       label: 'Edit',
       submenu: [
         {
           label: 'Cut',
           role: 'cut'
         },
         {
           label: 'Copy',
           role: 'copy'
         },
         {
           label: 'Paste',
           role: 'paste'
         }
       ]
     }
   ];

   // Function to create the main window
   function createMainWindow() {
     const mainWindow = new BrowserWindow();
     mainWindow.loadFile('index.html');

     // Create the menu from the template
     const menu = Menu.buildFromTemplate(menuTemplate);
     Menu.setApplicationMenu(menu);
   }

   app.whenReady().then(createMainWindow);