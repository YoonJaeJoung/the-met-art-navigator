export default function HomeView({ setView }) {
  return (
    <div className="home-view" style={{ padding: '60px 40px', overflowY: 'auto', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ maxWidth: '800px', width: '100%', background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '40px', boxShadow: 'var(--shadow-md)' }}>
        <h1 style={{ fontSize: '32px', marginBottom: '20px', color: 'var(--color-primary)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '40px' }}>🏛️</span> Met Art Navigator
        </h1>
        <p style={{ fontSize: '16px', lineHeight: '1.6', marginBottom: '30px', color: 'var(--color-text)' }}>
          Explore the Metropolitan Museum of Art's vast collection using advanced semantic search, cross-modal AI, and visual mapping. Discover artworks by concept, visual similarity, or simply browse the galleries interactively.
        </p>

        <h2 style={{ fontSize: '20px', marginTop: '30px', marginBottom: '20px', borderBottom: '1px solid var(--color-border)', paddingBottom: '10px' }}>Explore the App</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '40px' }}>
            <div className="stat-card hover-card" style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }} onClick={() => setView('gallery')}
                 onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }}
                 onMouseOut={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>🖼️ Gallery</h3>
                <p style={{ marginTop: '10px', fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>Browse artworks and filter by department. See where they are located on the museum map.</p>
            </div>
            <div className="stat-card hover-card" style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }} onClick={() => setView('training')}
                 onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }}
                 onMouseOut={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>🧠 Training</h3>
                <p style={{ marginTop: '10px', fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>Train the AI model using contrastive learning. Watch the loss go down in real-time!</p>
            </div>
            <div className="stat-card hover-card" style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }} onClick={() => setView('search')}
                 onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }}
                 onMouseOut={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none'; }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>🔍 Search</h3>
                <p style={{ marginTop: '10px', fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>Search for artworks using natural language or image upload. Find AI-ranked results instantly.</p>
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
