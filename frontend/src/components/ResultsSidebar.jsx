import { useState, useEffect } from 'react';

export default function ResultsSidebar({ results, selectedId, onSelect }) {
  const [activeTab, setActiveTab] = useState('semantic');
  
  const semantic = results?.semantic || [];
  const visual = results?.visual || [];
  const hasResults = semantic.length > 0 || visual.length > 0;

  // Auto-switch to the tab that actually has results if the current one is empty
  useEffect(() => {
    if (semantic.length === 0 && visual.length > 0 && activeTab === 'semantic') {
      setActiveTab('visual');
    } else if (visual.length === 0 && semantic.length > 0 && activeTab === 'visual') {
      setActiveTab('semantic');
    }
  }, [results, activeTab]);

  if (!hasResults) {
    return (
      <div className="results-list">
        <div className="empty-state">
          <span className="empty-icon">🏛️</span>
          <h3>No Results Yet</h3>
          <p>Search by text or image to find artworks and see their gallery locations on the map.</p>
        </div>
      </div>
    );
  }

  const items = activeTab === 'semantic' ? semantic : visual;

  return (
    <div className="results-sidebar">
      <div className="results-tabs">
        <button 
          className={`results-tab-btn ${activeTab === 'semantic' ? 'active' : ''} ${semantic.length === 0 ? 'disabled' : ''}`}
          onClick={() => semantic.length > 0 && setActiveTab('semantic')}
        >
          🧠 Semantic ({semantic.length})
        </button>
        <button 
          className={`results-tab-btn ${activeTab === 'visual' ? 'active' : ''} ${visual.length === 0 ? 'disabled' : ''}`}
          onClick={() => visual.length > 0 && setActiveTab('visual')}
        >
          🖼️ Visual ({visual.length})
        </button>
      </div>

      <div className="results-list">
        {items.length === 0 ? (
          <div className="empty-state mini">
            <p>No matches in this category.</p>
          </div>
        ) : (
          items.map((r) => (
            <div
              key={r.objectID}
              className={`result-card ${selectedId === r.objectID ? 'selected' : ''}`}
              onClick={() => onSelect(r)}
            >
              {r.primaryImageSmall ? (
                <img
                  className="result-thumb"
                  src={r.primaryImageSmall}
                  alt={r.title}
                  loading="lazy"
                />
              ) : (
                <div className="result-thumb" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px' }}>
                  🖼️
                </div>
              )}
              <div className="result-info">
                <div className="result-title" title={r.title}>{r.title || 'Untitled'}</div>
                <div className="result-artist">{r.artistDisplayName || 'Unknown Artist'}</div>
                <div className="result-meta">
                  {r.GalleryNumber && (
                    <span className="result-badge gallery">Gallery {r.GalleryNumber}</span>
                  )}
                  {r.floor && (
                    <span className="result-badge">Floor {r.floor}</span>
                  )}
                  <span className="result-score">{(r.score * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
