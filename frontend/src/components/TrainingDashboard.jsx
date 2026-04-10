import { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import version21Data from '../mockData.json';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8765';

export default function TrainingDashboard() {
  const [status, setStatus] = useState('idle'); // idle | training | complete | error
  const [lossData, setLossData] = useState([]);
  const [currentMetrics, setCurrentMetrics] = useState({
    epoch: 0,
    step: 0,
    loss: 0,
    temperature: 0,
  });
  const [serverStatus, setServerStatus] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(512);
  const [lr, setLr] = useState(0.0005);
  const [jointDim, setJointDim] = useState(512);
  const ws = useRef(null);

  const isLocal = typeof window !== 'undefined' && 
                  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

  // Load real version_21 data for presentation
  useEffect(() => {
    if (!isLocal) {
      setLossData(version21Data);
      setStatus('complete');
      
      const lastPoint = version21Data[version21Data.length - 1];
      setCurrentMetrics({ 
        epoch: lastPoint.epoch || 21, 
        step: lastPoint.step || 0, 
        loss: lastPoint.loss || 0, 
        val_loss: lastPoint.val_loss || 0,
        temperature: 0.05 
      });
      setServerStatus({ index_size: 27539, metadata_size: 27539 });
    }
  }, [isLocal]);

  // Fetch server status
  useEffect(() => {
    if (!isLocal) return;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API}/status`);
        const data = await res.json();
        setServerStatus(data);
        if (data.training_status === 'training') setStatus('training');
        else if (data.training_status === 'complete') setStatus('complete');
      } catch (e) {
        // Server not running
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket connection for live telemetry
  const connectWebSocket = useCallback(() => {
    if (ws.current) ws.current.close();
    const socket = new WebSocket(WS_URL);
    
    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'train_step') {
        setCurrentMetrics({
          epoch: msg.epoch,
          step: msg.step,
          loss: msg.loss,
          temperature: msg.temperature,
        });
        setLossData((prev) => {
          // Adaptive Downsampling: Ensure the entire run (up to totalExpectedSteps)
          // fits within a safe budget (~1000 points) to prevent SVG memory crashes.
          const skipRate = totalExpectedSteps > 1000 ? Math.ceil(totalExpectedSteps / 1000) : 1;
          
          if (msg.step % skipRate !== 0) return prev;
          
          const next = [...prev, { step: msg.step, loss: msg.loss, epoch: msg.epoch }];
          // No slicing needed anymore because adaptive skipRate keeps total points low
          return next;
        });
      } else if (msg.type === 'epoch_end') {
        setCurrentMetrics((prev) => ({ ...prev, val_loss: msg.val_loss }));
        setLossData((prev) => {
          if (prev.length === 0) return prev;
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], val_loss: msg.val_loss };
          return next;
        });
      } else if (msg.type === 'training_complete') {
        setStatus('complete');
      } else if (msg.type === 'training_start') {
        setStatus('training');
        setLossData([]);
      }
    };

    socket.onclose = () => {
      // Reconnect after 2s if training
      if (status === 'training') {
        setTimeout(connectWebSocket, 2000);
      }
    };

    ws.current = socket;
  }, [status]);

  useEffect(() => {
    if (status === 'training') {
      connectWebSocket();
    }
    return () => {
      if (ws.current) ws.current.close();
    };
  }, [status, connectWebSocket]);

  const startTraining = async () => {
    setStatus('training');
    setLossData([]);
    try {
      await fetch(`${API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, batch_size: batchSize, lr, joint_dim: jointDim }),
      });
      // Connect WebSocket for metrics
      setTimeout(connectWebSocket, 1000);
    } catch (err) {
      console.error('Failed to start training:', err);
      setStatus('error');
    }
  };

  const statusBadge = (
    <span className={`status-badge ${status}`}>
      <span className="status-dot" />
      {status === 'idle' && 'Idle'}
      {status === 'training' && 'Training...'}
      {status === 'complete' && 'Complete'}
      {status === 'error' && 'Error'}
    </span>
  );

  const stepsPerEpoch = serverStatus ? Math.ceil((serverStatus.metadata_size * 0.9) / batchSize) : 0;
  const totalExpectedSteps = epochs * stepsPerEpoch;

  return (
    <div className="training-dashboard">
      <div className="dashboard-header">
        <div>
          <h2>Model Training</h2>
          <div style={{ marginTop: 6 }}>{statusBadge}</div>
        </div>
        {isLocal ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
              Epochs:
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
              min={1}
              max={500}
              style={{
                width: 60,
                marginLeft: 6,
                padding: '4px 8px',
                background: 'var(--color-bg)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--color-text)',
                fontSize: 12,
                fontFamily: 'inherit',
              }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Batch:
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 256)}
              min={1}
              style={{
                width: 60, marginLeft: 6, padding: '4px 8px', background: 'var(--color-bg)',
                border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', color: 'var(--color-text)',
                fontSize: 12, fontFamily: 'inherit'
              }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            LR:
            <input
              type="number"
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
              step="0.0001"
              min="0"
              style={{
                width: 70, marginLeft: 6, padding: '4px 8px', background: 'var(--color-bg)',
                border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', color: 'var(--color-text)',
                fontSize: 12, fontFamily: 'inherit'
              }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Joint Dim:
            <input
              type="number"
              value={jointDim}
              onChange={(e) => setJointDim(parseInt(e.target.value) || 512)}
              min={16}
              style={{
                width: 60, marginLeft: 6, padding: '4px 8px', background: 'var(--color-bg)',
                border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', color: 'var(--color-text)',
                fontSize: 12, fontFamily: 'inherit'
              }}
            />
          </label>
          <button
            className="train-btn"
            onClick={startTraining}
            disabled={status === 'training'}
          >
            {status === 'training' ? (
              <>
                <span className="spinner" />
                Training
              </>
            ) : (
              '🚀 Start Training'
            )}
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px', background: 'var(--color-surface-raised)', borderRadius: '8px', border: '1px solid var(--color-border)' }}>
            <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>
              <strong>Demo Mode:</strong> Showing finalized <code>version_21</code> training profile (d_joint: 128, d_text: 768, lr: 0.0005).<br/>
              To actively train models on distinct parameters, please clone the project and run locally.
            </span>
          </div>
        )}
      </div>

      <div className="dashboard-cards">
        <div className="stat-card">
          <div className="stat-label">Epoch</div>
          <div className="stat-value">{currentMetrics.epoch + 1}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Step</div>
          <div className="stat-value">{currentMetrics.step}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Loss</div>
          <div className={`stat-value ${currentMetrics.loss < 1 ? 'success' : currentMetrics.loss < 3 ? 'warning' : 'error'}`}>
            {currentMetrics.loss.toFixed(4)}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Temperature</div>
          <div className="stat-value">{currentMetrics.temperature.toFixed(3)}</div>
        </div>
        {serverStatus && (
          <>
            <div className="stat-card">
              <div className="stat-label">Index Size</div>
              <div className="stat-value">{serverStatus.index_size.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Artworks</div>
              <div className="stat-value">{serverStatus.metadata_size.toLocaleString()}</div>
            </div>
          </>
        )}
      </div>

      <div className="chart-container">
        <h3>Training Loss over Steps</h3>
        {lossData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="step"
                domain={[0, totalExpectedSteps || 'auto']}
                type="number"
                tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--color-border)' }}
              />
              <YAxis
                tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--color-border)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--color-surface-raised)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                }}
                labelStyle={{ color: 'var(--color-text-secondary)' }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-primary)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="var(--color-warning)"
                strokeWidth={2}
                dot={true}
                connectNulls={true}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state" style={{ height: 300 }}>
            <span className="empty-icon">📊</span>
            <h3>No Training Data</h3>
            <p>Start training to see the live loss curve.</p>
          </div>
        )}
      </div>
    </div>
  );
}
