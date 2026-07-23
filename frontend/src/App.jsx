import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE = 'http://localhost:8000'

function TrustSeal({ score, reliability }) {
  return (
    <div className={`trust-seal seal-${reliability.toLowerCase()}`}>
      <div className="seal-inner">
        <span className="seal-score">{Math.round(score)}</span>
        <span className="seal-max">/100</span>
      </div>
      <span className="seal-label">{reliability}</span>
    </div>
  )
}

function TrustCard({ profile }) {
  const t = profile.trust
  return (
    <div className="card">
      <div className="card-top">
        <div>
          <h2>{profile.hotel_name}</h2>
          <p className="area">{profile.area}</p>
        </div>
        <TrustSeal score={t.trust_score} reliability={t.reliability} />
      </div>

      <div className="stat-row">
        <div className="stat">
          <span className="stat-value">{profile.avg_rating}/10</span>
          <span className="stat-label">Guest Rating</span>
        </div>
        <div className="stat">
          <span className="stat-value">{profile.review_count}</span>
          <span className="stat-label">Reviews Analyzed</span>
        </div>
      </div>

      <div className="breakdown">
        <h4>Trust Score Breakdown</h4>
        {Object.entries(t.breakdown).map(([key, value]) => (
          <div className="breakdown-row" key={key}>
            <span className="breakdown-label">{key.replace('_', ' ')}</span>
            <div className="breakdown-bar-bg">
              <div className="breakdown-bar-fill" style={{ width: `${(value / 35) * 100}%` }} />
            </div>
            <span className="breakdown-value">{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ResultsList({ results, onSelect, emptyMessage, showMatch, showDistance }) {
  if (!results || results.length === 0) {
    return <p className="empty-message">{emptyMessage}</p>
  }
  return (
    <div className="results-list">
      {results.map(r => (
        <button className="result-row" key={r.hotel_id} onClick={() => onSelect(r.hotel_id)}>
          <div className="result-main">
            <span className="result-name">{r.hotel_name}</span>
            <span className="result-area">{r.area}</span>
          </div>
          <div className="result-side">
            {showMatch && <span className="result-match">{r.match_pct}% match</span>}
            {showDistance && <span className="result-match">{r.approx_distance_km} km</span>}
            {r.matched_on && r.matched_on.length > 0 && (
              <span className="result-tags">{r.matched_on.join(', ')}</span>
            )}
          </div>
        </button>
      ))}
    </div>
  )
}

function AIWorkingSteps({ steps }) {
  return (
    <div className="ai-working">
      <div className="ai-working-header">Trusted by AI</div>
      <div className="ai-working-sub">Analyzing reviews...</div>
      <div className="ai-steps">
        {steps.map((step, i) => (
          <div className="ai-step" style={{ animationDelay: `${i * 0.4}s` }} key={step}>
            <span className="ai-step-check">✓</span> {step}
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const [meta, setMeta] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(false)

  const [mode, setMode] = useState('search') // 'search' | 'priorities' | 'nearby'

  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)

  // Priority matcher state
  const [priorityText, setPriorityText] = useState('')
  const [priorityResults, setPriorityResults] = useState(null)
  const [priorityLoading, setPriorityLoading] = useState(false)

  // Nearby-me state
  const [nearbyResults, setNearbyResults] = useState(null)
  const [nearbyLoading, setNearbyLoading] = useState(false)
  const [nearbyError, setNearbyError] = useState(null)

  function withMinDelay(promise, minMs) {
    const start = Date.now()
    return promise.then(result => {
      const elapsed = Date.now() - start
      const remaining = minMs - elapsed
      if (remaining > 0) {
        return new Promise(resolve => setTimeout(() => resolve(result), remaining))
      }
      return result
    })
  }

  useEffect(() => {
    axios.get(`${API_BASE}/meta`).then(res => setMeta(res.data))
  }, [])

  useEffect(() => {
    setSelectedId(null)
    setProfile(null)
  }, [mode])

  useEffect(() => {
    if (selectedId == null) return
    setLoading(true)
    axios.get(`${API_BASE}/hotels/${selectedId}`).then(res => {
      setProfile(res.data)
      setLoading(false)
    })
  }, [selectedId])

  // Debounced search-as-you-type
  useEffect(() => {
    if (searchQuery.trim().length < 2) {
      setSearchResults(null)
      setHasSearched(false)
      return
    }
    const timeout = setTimeout(() => {
      axios.get(`${API_BASE}/search`, { params: { q: searchQuery } }).then(res => {
        setSearchResults(res.data.results)
        setHasSearched(true)
      })
    }, 300)
    return () => clearTimeout(timeout)
  }, [searchQuery])

  function handlePrioritySubmit(e) {
    e.preventDefault()
    if (!priorityText.trim()) return
    setPriorityLoading(true)
    withMinDelay(axios.post(`${API_BASE}/match`, { text: priorityText }), 1800).then(res => {
      setPriorityResults(res.data)
      setPriorityLoading(false)
    })
  }

  function handleNearbyMe() {
    setNearbyError(null)
    setNearbyLoading(true)
    if (!navigator.geolocation) {
      setNearbyError('Location access is not supported in this browser.')
      setNearbyLoading(false)
      return
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords
        withMinDelay(axios.post(`${API_BASE}/nearby`, { latitude, longitude }), 1800).then(res => {
          setNearbyResults(res.data.results)
          setNearbyLoading(false)
        })
      },
      () => {
        setNearbyError('Location access was denied. Enable it in your browser settings to use this.')
        setNearbyLoading(false)
      }
    )
  }

  return (
    <div className="app">
      <header>
        <h1>InnSight</h1>
        <p className="tagline">See what other guests won't tell you</p>
        <p className="hero-sentence">
          AI analyzes real guest reviews to reveal which hotels are genuinely worth your money.
        </p>
        {meta && (
          <div className="stat-strip">
            <div className="stat-chip">
              <span className="stat-chip-value">{meta.hotel_count}</span>
              <span className="stat-chip-label">Hotels</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-value">{meta.review_count.toLocaleString()}</span>
              <span className="stat-chip-label">Reviews</span>
            </div>
            <div className="stat-chip">
              <span className="stat-chip-value">AI</span>
              <span className="stat-chip-label">Trust Scores</span>
            </div>
          </div>
        )}
      </header>

      <div className="mode-tabs">
        <button className={mode === 'search' ? 'tab active' : 'tab'} onClick={() => setMode('search')}>
          Search
        </button>
        <button className={mode === 'priorities' ? 'tab active' : 'tab'} onClick={() => setMode('priorities')}>
          Tell us what you want
        </button>
        <button className={mode === 'nearby' ? 'tab active' : 'tab'} onClick={() => setMode('nearby')}>
          Near me
        </button>
      </div>

      <div className={`workspace ${profile || loading ? 'has-result' : 'centered'}`}>
        <div className="panel-col">
          {mode === 'search' && (
            <div className="mode-panel">
              <input
                className="text-input"
                placeholder="Search hotel name or area (e.g. Paharganj)"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              {hasSearched && (
                <ResultsList
                  results={searchResults}
                  onSelect={setSelectedId}
                  showMatch={false}
                  emptyMessage={
                    meta
                      ? `No matches in our dataset. InnSight currently only covers ${meta.hotel_count} hotels in ${meta.city} — this hotel may be outside that coverage, or not registered in our data yet.`
                      : 'No matches found.'
                  }
                />
              )}
            </div>
          )}

          {mode === 'priorities' && (
            <div className="mode-panel">
              <form onSubmit={handlePrioritySubmit} className="priority-form">
                <textarea
                  className="text-input"
                  placeholder="e.g. I want a quiet hotel near the metro with good breakfast"
                  value={priorityText}
                  onChange={e => setPriorityText(e.target.value)}
                  rows={2}
                />
                <button type="submit" className="primary-button" disabled={priorityLoading}>
                  {priorityLoading ? 'Matching...' : 'Find hotels for me'}
                </button>
              </form>
              {priorityLoading && (
                <AIWorkingSteps steps={['Understanding your request', 'Extracting amenities', 'Ranking hotels']} />
              )}
              {priorityResults && !priorityLoading && (
                <>
                  {Object.keys(priorityResults.detected_priorities).length > 0 && (
                    <p className="detected-note">
                      Prioritizing: {Object.keys(priorityResults.detected_priorities).map(a => a.replace('_', ' ')).join(', ')}
                    </p>
                  )}
                  <ResultsList
                    results={priorityResults.results}
                    onSelect={setSelectedId}
                    showMatch={true}
                    emptyMessage="Couldn't find a strong match — try different words."
                  />
                </>
              )}
            </div>
          )}

          {mode === 'nearby' && (
            <div className="mode-panel">
              <button className="primary-button" onClick={handleNearbyMe} disabled={nearbyLoading}>
                {nearbyLoading ? 'Finding your location...' : 'Suggest hotels near me'}
              </button>
              {nearbyError && <p className="empty-message">{nearbyError}</p>}
              {nearbyLoading && (
                <AIWorkingSteps steps={['Detecting fake reviews', 'Matching nearby hotels', 'Ranking by distance']} />
              )}
              {nearbyResults && !nearbyLoading && (
                <>
                  <p className="detected-note">
                    Distances are approximate, based on locality, not exact address.
                  </p>
                  <ResultsList
                    results={nearbyResults}
                    onSelect={setSelectedId}
                    showDistance={true}
                    emptyMessage="No nearby hotels found."
                  />
                </>
              )}
            </div>
          )}
        </div>

        <div className="result-col">
          {loading && <p className="empty-message">Loading hotel profile...</p>}
          {profile && !loading && <TrustCard profile={profile} key={profile.hotel_id} />}
          {!profile && !loading && (
            <p className="empty-message placeholder-message">
              Select a hotel to see its Trust Score, reliability breakdown, and analysis.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default App