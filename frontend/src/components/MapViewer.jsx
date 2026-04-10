import { useMemo, useRef, useState } from 'react';

const FLOOR_CONFIG = [
  { id: '1', label: 'Floor 1', file: '/map/floor1.png' },
  { id: '1M', label: 'Floor 1M', file: '/map/floor1M.png' },
  { id: '2', label: 'Floor 2', file: '/map/floor2.png' },
  { id: '3', label: 'Floor 3', file: '/map/floor3.png' },
];

function PinIcon({ selected }) {
  return (
    <svg viewBox="0 0 24 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M12 0C5.372 0 0 5.372 0 12c0 9 12 24 12 24s12-15 12-24c0-6.628-5.372-12-12-12z"
        fill={selected ? '#f0c75e' : '#c8102e'}
      />
      <circle cx="12" cy="12" r="5" fill="white" />
    </svg>
  );
}

export default function MapViewer({ results, selectedResult, onSelectResult }) {
  const [activeFloor, setActiveFloor] = useState('1');
  const mapRef = useRef(null);

  // Count pins per floor
  const pinsByFloor = useMemo(() => {
    const counts = {};
    (results || []).forEach((r) => {
      if (r.floor) {
        counts[r.floor] = (counts[r.floor] || 0) + 1;
      }
    });
    return counts;
  }, [results]);

  // Auto-switch to the floor with most results  
  useMemo(() => {
    if (selectedResult?.floor) {
      setActiveFloor(selectedResult.floor);
    } else if (results?.length > 0) {
      const floors = Object.entries(pinsByFloor);
      if (floors.length > 0) {
        floors.sort((a, b) => b[1] - a[1]);
        setActiveFloor(floors[0][0]);
      }
    }
  }, [selectedResult, results, pinsByFloor]);

  const floorConfig = FLOOR_CONFIG.find((f) => f.id === activeFloor) || FLOOR_CONFIG[0];
  const floorPins = (results || []).filter((r) => r.floor === activeFloor && r.x_pct != null && r.y_pct != null);

  return (
    <div className="map-viewer">
      <div className="floor-tabs">
        {FLOOR_CONFIG.map((f) => (
          <button
            key={f.id}
            className={`floor-tab ${activeFloor === f.id ? 'active' : ''}`}
            onClick={() => setActiveFloor(f.id)}
          >
            {f.label}
            {pinsByFloor[f.id] > 0 && (
              <span className="pin-count">{pinsByFloor[f.id]}</span>
            )}
          </button>
        ))}
      </div>

      <div className="map-container">
        {results && results.length > 0 ? (
          <div className="map-wrapper" ref={mapRef}>
            <img
              src={`http://localhost:8000${floorConfig.file}`}
              alt={floorConfig.label}
              draggable={false}
            />
            {floorPins.map((pin) => (
              <div
                key={pin.objectID}
                className={`map-pin ${selectedResult?.objectID === pin.objectID ? 'selected' : ''}`}
                style={{
                  left: `${pin.x_pct * 100}%`,
                  top: `${pin.y_pct * 100}%`,
                  zIndex: selectedResult?.objectID === pin.objectID ? 100 : 5,
                }}
                onClick={() => onSelectResult(pin)}
              >
                <PinIcon selected={selectedResult?.objectID === pin.objectID} />
                <div 
                  className="pin-tooltip"
                  style={{
                    transform: pin.x_pct > 0.8 ? 'translateX(-100%)' : pin.x_pct < 0.2 ? 'translateX(0)' : 'translateX(-50%)',
                    left: pin.x_pct > 0.8 ? '0' : pin.x_pct < 0.2 ? '100%' : '50%',
                    marginLeft: pin.x_pct > 0.8 ? '-12px' : pin.x_pct < 0.2 ? '12px' : '0'
                  }}
                >
                  {pin.title} — Gallery {pin.GalleryNumber}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <span className="empty-icon">🗺️</span>
            <h3>Museum Floor Plan</h3>
            <p>Search for artworks to see their gallery locations pinned on the map.</p>
          </div>
        )}
      </div>
    </div>
  );
}
