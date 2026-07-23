const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { SerialPort } = require('serialport');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

app.use(express.static(__dirname));

// Configure this to match your Teensy's serial port
// In production, you might want to use SerialPort.list() to auto-detect
const portName = process.env.SERIAL_PORT || '/dev/ttyACM0';
let serialPort = null;

try {
    serialPort = new SerialPort({ path: portName, baudRate: 115200 }, (err) => {
        if (err) {
            console.warn(`[WARN] Serial port ${portName} not found. Waiting for connection...`);
        } else {
            console.log(`[INFO] Connected to ${portName}`);
        }
    });
} catch(e) {
    console.error(e);
}

// Binary Frame Parser state machine
let buffer = Buffer.alloc(0);
const FRAME_START = 0xAA;
const FRAME_END = 0xBB;

if (serialPort) {
    serialPort.on('data', (data) => {
        buffer = Buffer.concat([buffer, data]);

        // Process all complete frames in the buffer
        while (buffer.length > 0) {
            // Find start byte
            const startIndex = buffer.indexOf(FRAME_START);
            if (startIndex === -1) {
                // No start byte, discard buffer
                buffer = Buffer.alloc(0);
                break;
            }

            // Drop bytes before start byte
            if (startIndex > 0) {
                buffer = buffer.subarray(startIndex);
            }

            // Check if we have at least minimum frame size (Phase B raw override = 18 bytes)
            if (buffer.length < 18) {
                break; // Wait for more data
            }

            // Identify frame type by checking for FRAME_END
            let frameLength = 0;
            let isPhaseC = false;

            if (buffer.length >= 34 && buffer[33] === FRAME_END) {
                frameLength = 34; // Phase A/C (8 DOF)
                isPhaseC = true;
            } else if (buffer.length >= 18 && buffer[17] === FRAME_END) {
                frameLength = 18; // Phase B (Raw 4-channel)
            } else {
                // Keep looking or wait for more data if we haven't reached max possible frame size
                if (buffer.length >= 34) {
                    // Invalid frame, drop the start byte and continue searching
                    buffer = buffer.subarray(1);
                    continue;
                }
                break;
            }

            // Extract frame payload
            const payload = buffer.subarray(1, frameLength - 1);
            const floats = [];
            for (let i = 0; i < payload.length; i += 4) {
                floats.push(payload.readFloatLE(i));
            }

            if (isPhaseC && floats.length === 8) {
                // Broadcast 8-DOF ML state
                io.emit('telemetry', {
                    type: 'ml_state',
                    pitch: floats[0], yaw: floats[1], intensity: floats[2],
                    f1: floats[3], f2: floats[4], f3: floats[5],
                    arousal: floats[6], valence: floats[7]
                });
            } else if (!isPhaseC && floats.length === 4) {
                // Broadcast Raw DSP state
                io.emit('telemetry', {
                    type: 'raw_dsp',
                    a0: floats[0], a1: floats[1], a2: floats[2], a3: floats[3]
                });
            }

            // Move buffer forward
            buffer = buffer.subarray(frameLength);
        }
    });
}

io.on('connection', (socket) => {
    console.log('[INFO] Web client connected');
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`[INFO] CADENCE Relay Server running on http://localhost:${PORT}`);
});
