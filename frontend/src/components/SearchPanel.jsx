import { useState, useRef, useCallback } from 'react';

const API = import.meta.env.VITE_API_URL || 'https://ayjoung-met-art-navigator-api.hf.space';

export default function SearchPanel({ onResults, loading, setLoading }) {
  const [mode, setMode] = useState('text'); // 'text' | 'image'
  const [query, setQuery] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleTextSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/search/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: 20 }),
      });
      const data = await res.json();
      onResults(data.results || { semantic: [], visual: [] });
    } catch (err) {
      console.error('Search error:', err);
      onResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, onResults, setLoading]);

  const handleImageSearch = useCallback(async () => {
    if (!imageFile) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', imageFile);
      form.append('top_k', '20');
      const res = await fetch(`${API}/search/image`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      onResults(data.results || { semantic: [], visual: [] });
    } catch (err) {
      console.error('Search error:', err);
      onResults([]);
    } finally {
      setLoading(false);
    }
  }, [imageFile, onResults, setLoading]);

  const handleFileChange = (file) => {
    if (!file) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileChange(file);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleTextSearch();
  };

  return (
    <div className="search-panel">
      <div className="search-panel-header">
        <h2>🔍 Search</h2>
        <p>Find artworks by text or image</p>
      </div>

      <div className="search-mode-toggle">
        <button
          className={`mode-btn ${mode === 'text' ? 'active' : ''}`}
          onClick={() => setMode('text')}
        >
          ✏️ Text Query
        </button>
        <button
          className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
          onClick={() => setMode('image')}
        >
          🖼️ Image Upload
        </button>
      </div>

      <div className="search-input-area">
        {mode === 'text' ? (
          <div className="text-input-wrapper">
            <input
              id="search-text-input"
              type="text"
              placeholder="e.g. impressionist landscape, cubist still life..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              className="search-btn"
              onClick={handleTextSearch}
              disabled={loading || !query.trim()}
              title="Search"
            >
              →
            </button>
          </div>
        ) : (
          <>
            <div
              className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <span className="upload-icon">📤</span>
              <p>Drop an artwork image here or click to browse</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => handleFileChange(e.target.files[0])}
              />
            </div>

            {imagePreview && (
              <div className="upload-preview">
                <img src={imagePreview} alt="Upload preview" />
                <button
                  className="remove-btn"
                  onClick={() => { setImageFile(null); setImagePreview(null); }}
                >
                  ✕
                </button>
              </div>
            )}

            <button
              className="submit-search-btn"
              onClick={handleImageSearch}
              disabled={loading || !imageFile}
            >
              {loading ? 'Searching...' : 'Search by Image'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
