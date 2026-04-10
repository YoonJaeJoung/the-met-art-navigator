import { useState } from 'react';
import SearchPanel from './components/SearchPanel';
import ResultsSidebar from './components/ResultsSidebar';
import MapViewer from './components/MapViewer';
import TrainingDashboard from './components/TrainingDashboard';
import GalleryView from './components/GalleryView';
import ArtworkModal from './components/ArtworkModal';
import HomeView from './components/HomeView';
import './index.css';

export default function App() {
  const [view, setView] = useState('home'); // 'home' | 'search' | 'training' | 'gallery'
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [modalResult, setModalResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleResults = (newResults) => {
    setResults(newResults);
    setSelectedResult(newResults.length > 0 ? newResults[0] : null);
  };

  const handleSelectResult = (result) => {
    setSelectedResult(result);
    setModalResult(result);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>
          <span className="logo-icon">🏛️</span>
          Met Art Navigator
        </h1>
        <nav className="header-nav">
          <button
            className={`nav-tab ${view === 'home' ? 'active' : ''}`}
            onClick={() => setView('home')}
          >
            🏠 Home
          </button>
          <button
            className={`nav-tab ${view === 'gallery' ? 'active' : ''}`}
            onClick={() => setView('gallery')}
          >
            🖼️ Gallery
          </button>
          <button
            className={`nav-tab ${view === 'training' ? 'active' : ''}`}
            onClick={() => setView('training')}
          >
            🧠 Training
          </button>
          <button
            className={`nav-tab ${view === 'search' ? 'active' : ''}`}
            onClick={() => setView('search')}
          >
            🔍 Search
          </button>
        </nav>
      </header>

      {/* Body */}
      <div className="app-body">
        {view === 'home' ? (
          <HomeView setView={setView} />
        ) : view === 'search' ? (
          <>
            {/* Left: Search + Results */}
            <div style={{ display: 'flex', flexDirection: 'column', width: 360, flexShrink: 0, borderRight: '1px solid var(--color-border)' }}>
              <SearchPanel
                onResults={handleResults}
                loading={loading}
                setLoading={setLoading}
              />
              <ResultsSidebar
                results={results}
                selectedId={selectedResult?.objectID}
                onSelect={handleSelectResult}
              />
            </div>

            {/* Right: Map */}
            <MapViewer
              results={results}
              selectedResult={selectedResult}
              onSelectResult={handleSelectResult}
            />
          </>
        ) : view === 'training' ? (
          <TrainingDashboard />
        ) : (
          <GalleryView />
        )}
      </div>

      {modalResult && (
        <ArtworkModal 
          artwork={modalResult} 
          onClose={() => setModalResult(null)} 
        />
      )}
    </div>
  );
}
