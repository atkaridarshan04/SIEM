"use client";
import { useState, useEffect } from "react";
import Navbar from "@/app/components/navbar";
import { io } from 'socket.io-client';
import { toast, ToastContainer } from "react-toastify";
import { Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Home() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const logsResponse = await fetch('http://localhost:5000/api/logs');
        const logsData = await logsResponse.json();
        setLogs(logsData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    const socket = io('http://localhost:5000');

    socket.on('new_log', (data) => {
      setLogs(prev => [data, ...prev]);
      if (data.classification === 'threat') {
        toast.error(`New Threat Detected: ${data.log}`, {
          position: "top-right",
          theme: "dark",
          autoClose: 5000,
          pauseOnHover: true,
        });
      }
    });

    return () => socket.disconnect();
  }, []);

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'No timestamp';
    const date = new Date(timestamp);
    return isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleString();
  };

  // Group threats by classification
  const threatCounts = logs.reduce((acc, log) => {
    if (log.classification && log.classification !== "Normal") {
      acc[log.classification] = (acc[log.classification] || 0) + 1;
    }
    return acc;
  }, {});

  const labels = Object.keys(threatCounts);
  const dataValues = Object.values(threatCounts);

  const pieData = {
    labels,
    datasets: [
      {
        data: dataValues,
        backgroundColor: [
          '#ff4d4d',    // Memory Error - red
          '#ff9933',    // Authentication Error - orange
          '#ffcc00',    // File System Error - yellow
          '#3399ff',    // Network Error - blue
          '#cc33ff',    // Permission Error - purple
          '#00cc7d',    // Other - green
        ],
        hoverBackgroundColor: [
          '#ff6666',
          '#ffad5c',
          '#ffdb4d',
          '#66b3ff',
          '#d580ff',
          '#33d38a',
        ],
      },
    ],
  };

  const clearHistory = () => {
    toast.success("History cleared successfully!");
    setLogs([]);
  };

  return (
    <div className="min-h-screen bg-[#1a1f2c]">
      <ToastContainer />
      <Navbar />
      <main className="p-4 sm:p-6 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Recent Logs */}
          <div className="p-3 sm:p-4 border border-gray-700 rounded bg-[#2c3e50] shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-lg sm:text-xl text-[#00ff9d]">Recent Logs</h2>
            </div>
            {loading ? (
              <p className="text-gray-400">Loading...</p>
            ) : logs.length > 0 ? (
              <ul className="space-y-2 max-h-[300px] sm:max-h-[350px] overflow-auto">
                {logs.slice(0, 5).map((log, index) => (
                  <li key={index} className="p-2 bg-[#1a1f2c] rounded border border-gray-700">
                    <p className="text-xs sm:text-sm text-gray-300">{log.log}</p>
                    <div className="flex justify-between items-center mt-1">
                      <div>
                        <p className="text-xs italic text-gray-400">
                          Classification: {log.classification || "Normal"}
                        </p>
                        <span className="text-[10px] sm:text-xs text-gray-500">
                          {formatTimestamp(log.timestamp)}
                        </span>
                      </div>
                      <button className="text-[10px] sm:text-xs text-blue-500 hover:underline">
                        Solution
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-400">No recent logs to display</p>
            )}
          </div>

          {/* Pie Chart for Threat Types */}
          <div className="p-3 sm:p-4 border border-gray-700 rounded bg-[#2c3e50] shadow-lg">
            <h2 className="text-lg sm:text-xl mb-3 text-[#00ff9d]">Threat Types Distribution</h2>
            {loading ? (
              <p className="text-gray-400">Loading...</p>
            ) : labels.length > 0 ? (
              <div className="w-full flex justify-center items-center" style={{ height: '280px' }}>
                <Pie
                  data={pieData}
                  options={{
                    maintainAspectRatio: false,
                    responsive: true,
                    plugins: {
                      legend: {
                        position: 'right',
                        labels: {
                          boxWidth: 12,
                          padding: 10,
                          font: {
                            size: window.innerWidth < 768 ? 10 : 12
                          }
                        }
                      }
                    }
                  }}
                />
              </div>
            ) : (
              <p className="text-gray-400">No threats detected</p>
            )}
          </div>
        </div>

        <div className="flex justify-between items-center mt-4 sm:mt-6">
          <button
            onClick={clearHistory}
            className="px-3 sm:px-4 py-1.5 sm:py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors duration-200 flex items-center gap-2 text-xs sm:text-sm"
          >
            Clear History
          </button>
        </div>
      </main>
    </div>
  );
}
