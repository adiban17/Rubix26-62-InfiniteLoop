const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    maxHttpBufferSize: 1e8 // Increase buffer for image uploads
});

app.use(express.static(path.join(__dirname, 'public')));

// Structure: { socketId: { name, roll, logs: [], ... } }
let activeStudents = {};

io.on('connection', (socket) => {
    socket.emit('update-dashboard', Object.values(activeStudents));

    socket.on('student-connect', (data) => {
        activeStudents[socket.id] = {
            id: socket.id,
            name: data.name,
            roll: data.roll,
            sap: data.sap,
            startTime: new Date().toLocaleTimeString(),
            endTime: '-',
            riskScore: 'Normal',
            logs: [] 
        };
        io.emit('update-dashboard', Object.values(activeStudents));
    });

    // 3. Status Update (Enhanced for Evidence)
    socket.on('student-status-update', (payload) => {
        if (activeStudents[socket.id]) {
            const student = activeStudents[socket.id];
            
            // Handle both legacy (string) and new (object) payloads
            let statusText = payload;
            let evidenceImage = null;

            if (typeof payload === 'object' && payload !== null) {
                statusText = payload.status;
                evidenceImage = payload.image;
            }

            student.riskScore = statusText;

            if (statusText.includes("VIOLATION")) {
                const logEntry = {
                    time: new Date().toLocaleTimeString(),
                    violation: statusText,
                    evidence: evidenceImage // Store base64 image or null
                };
                student.logs.push(logEntry);
            }

            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });

    socket.on('disconnect', () => {
        if (activeStudents[socket.id]) {
            activeStudents[socket.id].endTime = new Date().toLocaleTimeString();
            activeStudents[socket.id].riskScore = 'Offline';
            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});