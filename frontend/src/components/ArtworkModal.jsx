import React from 'react';

export default function ArtworkModal({ artwork, onClose }) {
  if (!artwork) return null;

  return (
    <div 
      className="modal-overlay" 
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '24px'
      }}
    >
      <div 
        className="modal-content" 
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--color-bg)',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-lg)',
          width: '100%',
          maxWidth: '800px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 24px 48px rgba(0,0,0,0.5)'
        }}
      >
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--color-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '20px', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {artwork.title || 'Untitled'}
          </h2>
          <button 
            onClick={onClose} 
            style={{ 
              background: 'transparent', border: 'none', color: 'var(--color-text-secondary)', 
              fontSize: '24px', cursor: 'pointer', padding: '0 8px' 
            }}
          >
            &times;
          </button>
        </div>
        
        <div style={{ display: 'flex', overflowY: 'auto', flexDirection: 'row', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 300px', borderRight: '1px solid var(--color-border)', background: 'var(--color-surface)' }}>
            {artwork.primaryImage || artwork.primaryImageSmall ? (
              <img 
                src={artwork.primaryImage || artwork.primaryImageSmall} 
                alt={artwork.title}
                style={{ width: '100%', height: 'auto', display: 'block', maxHeight: '600px', objectFit: 'contain' }}
              />
            ) : (
              <div style={{ width: '100%', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '48px' }}>
                🖼️
              </div>
            )}
          </div>
          
          <div style={{ flex: '1 1 350px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '4px' }}>{artwork.artistDisplayName || 'Unknown Artist'}</h3>
              <p style={{ color: 'var(--color-text-secondary)', fontSize: '14px' }}>
                {[artwork.culture, artwork.period].filter(Boolean).join(', ')}
              </p>
            </div>
            
            {(artwork.department || artwork.medium) && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {artwork.department && <span className="result-badge">{artwork.department}</span>}
                {artwork.medium && <span className="result-badge" style={{ background: 'var(--color-surface-raised)' }}>{artwork.medium}</span>}
                {artwork.GalleryNumber && <span className="result-badge gallery">Gallery {artwork.GalleryNumber}</span>}
              </div>
            )}

            {artwork.description && (
              <div>
                <h4 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--color-text-secondary)', marginBottom: '8px', letterSpacing: '0.05em' }}>Description</h4>
                <p style={{ fontSize: '14px', lineHeight: '1.6', color: 'var(--color-text-muted)' }}>
                  {artwork.description}
                </p>
              </div>
            )}
            
            {artwork.objectURL && (
              <div style={{ marginTop: 'auto', paddingTop: '16px' }}>
                <a 
                  href={artwork.objectURL} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="train-btn"
                  style={{ display: 'inline-block', textDecoration: 'none', textAlign: 'center' }}
                >
                  View on Met Website
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
