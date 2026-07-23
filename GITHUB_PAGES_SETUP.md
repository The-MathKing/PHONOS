# Hosting CADENCE on GitHub Pages

This guide explains how to host the CADENCE Diagnostic Web Application on GitHub Pages. 

## The Architecture
Because the Phase A/B/C system now utilizes high-speed **Binary USB Telemetry**, a local Node.js Relay Server is required to decode the binary bytes and translate them into a WebSocket JSON stream. 

GitHub Pages is a static hosting provider, meaning it cannot run the Node.js server. However, you can host the `index.html` and `app.js` on GitHub Pages, and run the Node.js Relay Server locally on your laptop!

When you visit your GitHub Pages URL, the 3D visualization will securely connect to `http://localhost:3000` on your machine to receive the live data from the Teensy.

---

## Step 1: Enable GitHub Pages
1. Go to your repository on GitHub: [PHONOS Repository](https://github.com/The-MathKing/PHONOS)
2. Click on **Settings** (the gear icon near the top right).
3. On the left sidebar, click on **Pages**.
4. Under the **Build and deployment** section:
   - Source: Select **Deploy from a branch**.
   - Branch: Select `main` and the `/` (root) folder.
   - Click **Save**.
5. Wait a few minutes. GitHub will provide you with a live URL (e.g., `https://the-mathking.github.io/PHONOS/`).

To access the diagnostic app, simply append `/diagnostic-app/` to the URL.
Example: `https://the-mathking.github.io/PHONOS/diagnostic-app/`

---

## Step 2: Configure `app.js` for Remote Hosting
Currently, `app.js` connects to the websocket using a relative path. We need to explicitly point it to your local machine for when it runs on the internet.

1. Open `diagnostic-app/app.js`.
2. Change the Socket.io initialization at the bottom from:
   ```javascript
   const socket = io();
   ```
   To:
   ```javascript
   const socket = io("http://localhost:3000");
   ```
3. Commit and push this change to GitHub.

---

## Step 3: Run the System
Whenever you want to use the neuro-prosthetic:

1. **Plug in the Teensy 4.1.**
2. **Start the local Relay Server:**
   Open your terminal on your laptop, navigate to the `diagnostic-app` folder, and run:
   ```bash
   npm install
   node server.js
   ```
3. **Open the Visualization:**
   Navigate to your live GitHub Pages URL in your browser (`https://the-mathking.github.io/PHONOS/diagnostic-app/`).
   
The beautiful 3D Particle Aura and live telemetry will instantly appear, driven by your local hardware!
