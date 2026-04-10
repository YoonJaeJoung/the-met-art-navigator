export default function ResultsSidebar({ results, selectedId, onSelect }) {
  if (!results || results.length === 0) {
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

  return (
    <div className="results-list">
      <div className="results-header">
        {results.length} result{results.length !== 1 ? 's' : ''} found
      </div>
      {results.map((r) => (
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
      ))}
    </div>
  );
}
