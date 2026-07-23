const { SerialPort } = require('serialport');
const { WebSocketServer } = require('ws');

// Update port to match your OS (e.g., /dev/ttyACM0 for Linux/Mac, COM3 for Windows)
const portName = process.env.SERIAL_PORT || '/dev/ttyACM0';
const wss = new WebSocketServer({ port: 8080 });

let serialPort = null;
try {
    serialPort = new SerialPort({ path: portName, baudRate: 115200 }, (err) => {
        if (err) {
            console.warn(`[WARN] Serial connection failed: ${err.message}`);
        } else {
            console.log(`[INFO] Connected to ${portName}`);
        }
    });
} catch(e) {
    console.error(e);
}

let buffer = Buffer.alloc(0);
const PACKET_SIZE = 34; // 1 start + (8 floats * 4 bytes) + 1 stop byte

if (serialPort) {
    serialPort.on('data', (data) => {
        buffer = Buffer.concat([buffer, data]);
        
        while (buffer.length >= PACKET_SIZE) {
            if (buffer[0] === 0xAA && buffer[PACKET_SIZE - 1] === 0xBB) {
                const phoneticVector = [];
                for (let i = 0; i < 6; i++) {
                    phoneticVector.push(buffer.readFloatLE(1 + (i * 4)));
                }
                const arousal = buffer.readFloatLE(25);
                const valence = buffer.readFloatLE(29);
                
                const payload = JSON.stringify({ phoneticVector, affect: { arousal, valence } });
                
                wss.clients.forEach(client => {
                    if (client.readyState === 1) client.send(payload);
                });
                
                buffer = buffer.subarray(PACKET_SIZE);
            } else {
                // Find next start delimiter byte if out of sync
                const nextStart = buffer.indexOf(0xAA, 1);
                if (nextStart !== -1) {
                    buffer = buffer.subarray(nextStart);
                } else {
                    buffer = Buffer.alloc(0);
                    break;
                }
            }
        }
    });
}

console.log("[INFO] Fast-Parser WebSocket Server running on ws://localhost:8080");
