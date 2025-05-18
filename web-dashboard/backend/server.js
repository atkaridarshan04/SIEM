const express = require('express');
const http = require('http');
const cors = require('cors');
const { Server } = require('socket.io');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const os = require('os');
const osUtils = require('os-utils');
const checkDiskSpace = require('check-disk-space').default;

const app = express();
const server = http.createServer(app);

// Setup Socket.IO
const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(cors());
app.use(bodyParser.json());

// MongoDB Connection
mongoose.connect('mongodb://localhost:27017/siem_logs')
  .then(() => {
    console.log('✅ Connected to MongoDB');
  })
  .catch((err) => {
    console.error('❌ MongoDB connection error:', err);
  });

// Allowed classifications
const allowedClasses = [
  'Normal',
  'Memory Error',
  'Authentication Error',
  'File System Error',
  'Network Error',
  'Permission Error'
];

// Mongoose Schema and Model
const logSchema = new mongoose.Schema({
  log: String,
  classification: {
    type: String,
    enum: allowedClasses,
    required: true
  },
  timestamp: { type: Date, default: Date.now }
});
const Log = mongoose.model('Log', logSchema);

// WebSocket connection
io.on('connection', (socket) => {
  console.log('🔌 Frontend connected:', socket.id);

  socket.on('disconnect', () => {
    console.log('❌ Frontend disconnected:', socket.id);
  });
});

// GET endpoint to fetch logs
app.get('/api/logs', async (req, res) => {
  const { classification } = req.query;

  try {
    const filter = classification ? { classification } : {};
    const logs = await Log.find(filter).sort({ timestamp: -1 }).limit(100);
    res.json(logs);
  } catch (err) {
    console.error('❌ Error fetching logs:', err);
    res.status(500).send('Failed to fetch logs from database');
  }
});

// POST endpoint to receive logs
app.post('/api/logs', async (req, res) => {
  const { log, classification } = req.body;

  if (!allowedClasses.includes(classification)) {
    return res.status(400).send('❌ Invalid classification type');
  }

  console.log('📥 Received log:', log, '| Classification:', classification);

  try {
    const newLog = new Log({ log, classification });
    await newLog.save();
    console.log('💾 Log saved to DB');

    io.emit('new_log', newLog);

    res.status(200).send('Log received and broadcasted');
  } catch (err) {
    console.error('❌ Error saving log:', err);
    res.status(500).send('Failed to save log to database');
  }
});

// GET endpoint for System Health
app.get('/api/system-health', async (req, res) => {
  const diskPath = os.platform() === 'win32' ? 'C:\\' : '/';

  osUtils.cpuUsage(async cpuPercent => {
    const memoryFree = os.freemem();
    const memoryTotal = os.totalmem();
    const memoryUsed = memoryTotal - memoryFree;

    try {
      const disk = await checkDiskSpace(diskPath);

      const data = {
        cpu: {
          usage: +(cpuPercent * 100).toFixed(2)
        },
        memory: {
          total: memoryTotal,
          free: memoryFree,
          used: memoryUsed,
          usedPercentage: +((memoryUsed / memoryTotal) * 100).toFixed(2)
        },
        disk: {
          total: +(disk.size / (1024 ** 3)).toFixed(2),
          free: +(disk.free / (1024 ** 3)).toFixed(2),
          used: +((disk.size - disk.free) / (1024 ** 3)).toFixed(2),
          usedPercentage: +(((disk.size - disk.free) / disk.size) * 100).toFixed(2)
        },
        system: {
          hostname: os.hostname(),
          uptime: os.uptime(),
          platform: os.platform(),
          arch: os.arch(),
          cores: os.cpus().length
        }
      };

      res.json(data);
    } catch (err) {
      console.error('❌ Disk check error:', err);
      res.status(500).json({ error: 'Failed to fetch disk usage' });
    }
  });
});

// Start server
server.listen(5000, () => {
  console.log('🚀 Express backend running on http://localhost:5000');
});
