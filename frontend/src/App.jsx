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
  const [results, setResults] = useState({ semantic: [], visual: [] });
  const [selectedResult, setSelectedResult] = useState(null);
  const [modalResult, setModalResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleResults = (newResults) => {
    setResults(newResults);
    // Pick first result for auto-selection, handling both flat array and categorized object
    let first = null;
    if (Array.isArray(newResults)) {
      first = newResults.length > 0 ? newResults[0] : null;
    } else if (newResults && typeof newResults === 'object') {
      const all = [...(newResults.semantic || []), ...(newResults.visual || [])];
      first = all.length > 0 ? all[0] : null;
    }
    setSelectedResult(first);
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
            className={`nav-tab ${view === 'search' ? 'active' : ''}`}
            onClick={() => setView('search')}
          >
            🔍 Search
          </button>
        </nav>
      </header>

      {/* Body */}
      <div className={`app-body view-${view} ${results && (results.semantic?.length > 0 || results.visual?.length > 0) ? 'has-results' : ''}`}>
        {view === 'home' ? (
          <HomeView setView={setView} />
        ) : view === 'search' ? (
          <>
            {/* Search + Results */}
            <div className="search-results-column">
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

            {/* Map */}
            <MapViewer
              results={
                Array.isArray(results) 
                  ? results 
                  : (results && (results.semantic || results.visual) 
                      ? [...(results.semantic || []), ...(results.visual || [])] 
                      : [])
              }
              selectedResult={selectedResult}
              onSelectResult={handleSelectResult}
            />
          </>
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
