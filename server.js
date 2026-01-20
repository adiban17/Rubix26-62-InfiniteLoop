const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { maxHttpBufferSize: 1e8 });

app.use(express.static(path.join(__dirname, 'public')));

let activeStudents = {};

// --- CONFIGURATION: RISK WEIGHTS ---
const RISK_VALUES = {
    TAB_SWITCH: 90,
    PHONE: 85,
    FACE_ISSUE: 75, // Covers Multiple Faces & No Face
    LOOKING: 40,
    NORMAL: 0
};

// --- HELPER: CALCULATE RISK FROM STATUS TEXT ---
function getInstantRisk(statusText) {
    if (!statusText) return RISK_VALUES.NORMAL;
    
    const upper = statusText.toUpperCase();
    console.log(`[DEBUG] Analyzing Status: "${upper}"`); // Debug log

    // 1. Tab Switching (90%) - Highest Priority
    // Detects "VIOLATION: CHROME", "VIOLATION: SAFARI", etc.
    if (upper.includes("VIOLATION") && 
        !upper.includes("LOOKING") && 
        !upper.includes("GAZE") && 
        !upper.includes("PHONE") && 
        !upper.includes("FACE") && 
        !upper.includes("DEVICE")) {
        return RISK_VALUES.TAB_SWITCH;
    }

    // 2. Phone Detection (85%)
    if (upper.includes("PHONE") || upper.includes("DEVICE")) {
        return RISK_VALUES.PHONE;
    }

    // 3. Face Issues (75%)
    if (upper.includes("MULTIPLE") || upper.includes("NO FACE") || upper.includes("FACE COUNT")) {
        return RISK_VALUES.FACE_ISSUE;
    }

    // 4. Looking Away / Gaze (40%)
    if (upper.includes("LOOKING") || upper.includes("GAZE") || upper.includes("SUSPICIOUS")) {
        return RISK_VALUES.LOOKING;
    }

    return RISK_VALUES.NORMAL;
}

io.on('connection', (socket) => {
    socket.emit('update-dashboard', Object.values(activeStudents));

    // --- NEW STUDENT CONNECTS ---
    socket.on('student-connect', (data) => {
        const now = new Date().toLocaleTimeString();
        console.log(`Student Connected: ${data.name}`);
        
        activeStudents[socket.id] = {
            id: socket.id,
            name: data.name,
            roll: data.roll,
            sap: data.sap,
            startTime: now,
            endTime: '-',
            status: 'SYSTEM: ACTIVE',
            riskScore: 0, 
            peakRisk: 0,  // Tracks the highest score of the session
            riskHistory: [{ t: now, y: 0 }],
            logs: [] 
        };
        io.emit('update-dashboard', Object.values(activeStudents));
    });

    // --- STATUS UPDATE FROM PYTHON APP ---
    socket.on('student-status-update', (payload) => {
        if (activeStudents[socket.id]) {
            const student = activeStudents[socket.id];
            const now = new Date().toLocaleTimeString();
            
            let statusText = "NORMAL";
            let evidenceImage = null;

            // Handle different payload structures safely
            if (typeof payload === 'string') {
                statusText = payload;
            } else if (typeof payload === 'object' && payload !== null) {
                statusText = payload.status || "NORMAL";
                evidenceImage = payload.image;
            }

            // 1. Calculate Score
            const instantScore = getInstantRisk(statusText);
            
            // 2. Update Student State
            student.status = statusText;
            student.riskScore = instantScore; 

            // 3. Update PEAK Risk (This drives the speedometer)
            if (instantScore > student.peakRisk) {
                student.peakRisk = instantScore;
            }

            // 4. Update Graph History (Limit to 50 points)
            const lastPoint = student.riskHistory[student.riskHistory.length - 1];
            if (!lastPoint || lastPoint.y !== instantScore || student.riskHistory.length === 1) {
                student.riskHistory.push({ t: now, y: instantScore });
                if (student.riskHistory.length > 50) student.riskHistory.shift();
            }

            // 5. Logging (Save violation to list)
            if (instantScore > 0) {
                const lastLog = student.logs[student.logs.length - 1];
                // Only log if it's a new violation type
                if (!lastLog || lastLog.violation !== statusText) {
                    console.log(`[VIOLATION] ${student.name}: ${statusText} (${instantScore}%)`);
                    student.logs.push({
                        time: now,
                        violation: statusText,
                        evidence: evidenceImage,
                        riskVal: instantScore // This fixes the "undefined" error
                    });
                }
            }

            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });

    socket.on('disconnect', () => {
        if (activeStudents[socket.id]) {
            activeStudents[socket.id].endTime = new Date().toLocaleTimeString();
            activeStudents[socket.id].status = 'Offline';
            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});