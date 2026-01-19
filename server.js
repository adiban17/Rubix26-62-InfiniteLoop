const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static(path.join(__dirname, 'public')));

// Store student data
let activeStudents = {};

io.on('connection', (socket) => {
    // 1. Send current data to any new admin
    socket.emit('update-dashboard', Object.values(activeStudents));

    // 2. Handle connection from Python Script (Student)
    socket.on('student-connect', (data) => {
        console.log(`Student Connected: ${data.name}`);
        activeStudents[socket.id] = {
            id: socket.id,
            name: data.name,
            roll: data.roll,
            sap: data.sap,
            startTime: new Date().toLocaleTimeString(),
            endTime: '-',
            riskScore: 'Normal' // Default status
        };
        io.emit('update-dashboard', Object.values(activeStudents));
    });

    // 3. Handle Status Updates from Python (e.g., "Looking Away")
    socket.on('student-status-update', (statusText) => {
        if (activeStudents[socket.id]) {
            // Only update if status changed to avoid flickering
            activeStudents[socket.id].riskScore = statusText;
            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });

    // 4. Handle Disconnect
    socket.on('disconnect', () => {
        if (activeStudents[socket.id]) {
            activeStudents[socket.id].endTime = new Date().toLocaleTimeString();
            activeStudents[socket.id].riskScore = 'Offline';
            io.emit('update-dashboard', Object.values(activeStudents));
            // Optional: delete activeStudents[socket.id];
        }
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});