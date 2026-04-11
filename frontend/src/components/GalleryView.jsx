import { useState, useEffect } from 'react';
import ArtworkModal from './ArtworkModal';
import MapViewer from './MapViewer';

const API = import.meta.env.VITE_API_URL || 'https://ayjoung-met-art-navigator-api.hf.space';

export default function GalleryView() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [modalResult, setModalResult] = useState(null);
  
  // Interactive Map States
  const [galleries, setGalleries] = useState([]);
  const [selectedGallery, setSelectedGallery] = useState(null);

  // Fetch full gallery map payload once
  useEffect(() => {
    const fetchMap = async () => {
      try {
        const res = await fetch(`${API}/gallery-map`);
        const data = await res.json();
        // Convert map dictionary to array for MapViewer consumption
        const mappedArray = Object.entries(data).map(([k, v]) => ({
          GalleryNumber: k,
          ...v
        }));
        setGalleries(mappedArray);
      } catch (err) {
        console.error('Failed to fetch gallery map:', err);
      }
    };
    fetchMap();
  }, []);

  useEffect(() => {
    const fetchGallery = async () => {
      setLoading(true);
      try {
        const url = new URL(`${API}/gallery`);
        url.searchParams.append('page', page);
        url.searchParams.append('page_size', '50');
        if (selectedGallery) {
          url.searchParams.append('gallery', selectedGallery);
        }

        const res = await fetch(url.toString());
        const data = await res.json();
        setItems(data.results || []);
        setTotalPages(data.pages || 1);
        setTotalCount(data.total || 0);
      } catch (err) {
        console.error('Failed to fetch gallery:', err);
      } finally {
        setTimeout(() => setLoading(false), 300);
      }
    };
    fetchGallery();
  }, [page, selectedGallery]);

  if (loading) {
    return (
      <div className="gallery-view">
        <div className="loading-state">
          <span className="spinner" />
          <p>Loading gallery items...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="gallery-view" style={{ flex: 1, padding: 24, overflowY: 'auto' }}>
      
      {/* Visual Map Layout Block */}
      <div style={{ marginBottom: 32, background: 'var(--color-surface-raised)', borderRadius: 12, padding: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0, fontSize: 18 }}>Museum Map Overview</h3>
          {selectedGallery && (
            <button 
              onClick={() => { setSelectedGallery(null); setPage(1); }}
              style={{ background: 'var(--color-surface)', padding: '6px 12px', border: '1px solid var(--color-border)', borderRadius: 6, cursor: 'pointer' }}
            >
              Clear Map Filter
            </button>
          )}
        </div>
        <div style={{ maxHeight: '60vh', overflow: 'auto', borderRadius: 8, border: '1px solid var(--color-border)' }}>
          <MapViewer 
            mode="galleries"
            results={galleries}
            selectedResult={selectedGallery}
            onSelectResult={(galleryId) => {
              if (selectedGallery === galleryId) {
                setSelectedGallery(null);
              } else {
                setSelectedGallery(galleryId);
              }
              setPage(1);
            }}
          />
        </div>
      </div>

      <div className="dashboard-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>{selectedGallery ? `Artworks in Gallery ${selectedGallery}` : 'All Processed Artworks'}</h2>
          <p style={{ color: 'var(--color-text-secondary)', marginTop: 4 }}>
            {totalCount} total artworks {selectedGallery ? 'found' : 'available'} (Page {page} of {totalPages}).
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button 
            className="train-btn" 
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{ padding: '6px 12px', background: 'var(--color-surface-raised)', color: 'var(--color-text)' }}
          >
            Previous
          </button>
          <span style={{ fontSize: 13, color: 'var(--color-text-secondary)', minWidth: '60px', textAlign: 'center' }}>Page {page}</span>
          <button 
            className="train-btn" 
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages || totalPages === 0}
            style={{ padding: '6px 12px', background: 'var(--color-surface-raised)', color: 'var(--color-text)' }}
          >
            Next
          </button>
        </div>
      </div>

      {!loading && items.length > 0 ? (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: 16,
          marginTop: 24
        }}>
          {items.map((item) => (
            <div 
              key={item.objectID} 
              className="stat-card" 
              style={{ display: 'flex', flexDirection: 'column', gap: 12, cursor: 'pointer', transition: 'transform 0.15s ease' }}
              onClick={() => setModalResult(item)}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >
              <div 
                style={{ 
                  height: 150, 
                  background: 'var(--color-surface-raised)',
                  backgroundImage: item.primaryImageSmall ? `url(${item.primaryImageSmall})` : 'none',
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  borderRadius: 'var(--radius-sm)'
                }}
              />
              <div>
                <h4 style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.title || 'Untitled'}
                </h4>
                <p style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 2 }}>
                  {item.artistDisplayName || 'Unknown Artist'}
                </p>
                <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {item.GalleryNumber && (
                    <span className="result-badge gallery">Gallery {item.GalleryNumber}</span>
                  )}
                  {item.department && (
                    <span className="result-badge">{item.department}</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : !loading ? (
        <div className="empty-state">
          <span className="empty-icon">🖼️</span>
          <h3>No Artworks Found</h3>
          <p>{selectedGallery ? `No processed artworks mapped to Gallery ${selectedGallery}.` : 'The metadata table is empty. Try running the data ingestion pipeline.'}</p>
        </div>
      ) : null}

      {modalResult && (
        <ArtworkModal 
          artwork={modalResult} 
          onClose={() => setModalResult(null)} 
        />
      )}
    </div>
  );
}
