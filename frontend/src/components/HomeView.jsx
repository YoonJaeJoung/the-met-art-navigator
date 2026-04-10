export default function HomeView({ setView }) {
  return (
    <div className="home-view" style={{ padding: '60px 40px', overflowY: 'auto', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ maxWidth: '800px', width: '100%', background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '40px', boxShadow: 'var(--shadow-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h1 style={{ fontSize: '32px', margin: 0, color: 'var(--color-primary)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '40px' }}>🏛️</span> Met Art Navigator
          </h1>
          <a 
            href="https://github.com/YoonJaeJoung/the-met-art-navigator" 
            target="_blank" 
            rel="noopener noreferrer"
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '6px', 
              fontSize: '14px', 
              color: 'var(--color-text-secondary)', 
              textDecoration: 'none',
              padding: '6px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--color-surface-raised)',
              border: '1px solid var(--color-border)',
              transition: 'var(--transition)'
            }}
            onMouseOver={e => { e.currentTarget.style.color = 'var(--color-primary)'; e.currentTarget.style.borderColor = 'var(--color-primary-muted)'; }}
            onMouseOut={e => { e.currentTarget.style.color = 'var(--color-text-secondary)'; e.currentTarget.style.borderColor = 'var(--color-border)'; }}
          >
            <svg height="20" viewBox="0 0 16 16" width="20" style={{ fill: 'currentColor' }}>
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            GitHub
          </a>
        </div>
        <p style={{ fontSize: '16px', lineHeight: '1.6', marginBottom: '30px', color: 'var(--color-text)' }}>
          Explore the Metropolitan Museum of Art's vast collection using advanced semantic search, cross-modal AI, and visual mapping. Discover artworks by concept, visual similarity, or simply browse the galleries interactively.
        </p>

        <h2 style={{ fontSize: '20px', marginTop: '30px', marginBottom: '20px', borderBottom: '1px solid var(--color-border)', paddingBottom: '10px' }}>Explore the App</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '40px' }}>
            <div className="stat-card hover-card" style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }} onClick={() => setView('gallery')}
                 onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }}
                 onMouseOut={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>🖼️ Gallery</h3>
                <p style={{ marginTop: '10px', fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>Browse artworks and filter by department. See where they are located on the museum map with interactive pins.</p>
            </div>
            <div className="stat-card hover-card" style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }} onClick={() => setView('search')}
                 onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }}
                 onMouseOut={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>🔍 Dual Search</h3>
                <p style={{ marginTop: '10px', fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                  <strong>Semantic Search</strong>: Targeted matching for artworks with detailed descriptions. <br/>
                  <strong>Visual Search</strong>: Uses cross-modal learning to find artworks without descriptions based on their visual features.
                </p>
            </div>
        </div>

        <h2 style={{ fontSize: '20px', marginTop: '30px', marginBottom: '15px', borderBottom: '1px solid var(--color-border)', paddingBottom: '10px' }}>Data & Architecture</h2>
        <ul style={{ lineHeight: '1.8', marginLeft: '20px', color: 'var(--color-text-secondary)', fontSize: '14px' }}>
          <li><strong>Rich Dataset</strong>: Includes thousands of verified artworks from the Met collection.</li>
          <li><strong>Cross-Modal AI</strong>: Embeddings are generated using DINOv2 for images and Nomic Embed Text for descriptions, mapped into a joint semantic space.</li>
          <li><strong>Interactive Floor Plans</strong>: Support multi-floor gallery tracking with dynamic tooltip routing.</li>
          <li><strong>Live Telemetry</strong>: Websocket-powered dashboard visualizes PyTorch Lightning training metrics.</li>
        </ul>
      </div>
    </div>
  );
}
