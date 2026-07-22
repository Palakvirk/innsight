import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE = 'http://localhost:8000'

function TrustCard({ profile }) {
  const t = profile.trust
  return (
    <div className="card">
      <h2>{profile.hotel_name}</h2>
      <p className="area">{profile.area}</p>
      <div className="trust-header">
        <div className="stat">
          <span className="stat-value">{profile.avg_rating}/10</span>
          <span className="stat-label">Guest Rating</span>
        </div>
        <div className="stat">
          <span className="stat-value">{t.trust_score}/100</span>
          <span className="stat-label">AI Trust Score</span>
        </div>
        <div className={`reliability reliability-${t.reliability.toLowerCase()}`}>
          {t.reliability} Reliability
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

function App() {
  const [hotels, setHotels] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    axios.get(`${API_BASE}/hotels`).then(res => {
      setHotels(res.data)
      if (res.data.length > 0) setSelectedId(res.data[0].hotel_id)
    })
  }, [])

  useEffect(() => {
    if (selectedId == null) return
    setLoading(true)
    axios.get(`${API_BASE}/hotels/${selectedId}`).then(res => {
      setProfile(res.data)
      setLoading(false)
    })
  }, [selectedId])

  return (
    <div className="app">
      <header>
        <h1>InnSight</h1>
        <p className="tagline">See what other guests won't tell you</p>
      </header>

      <select
        className="hotel-picker"
        value={selectedId ?? ''}
        onChange={e => setSelectedId(Number(e.target.value))}
      >
        {hotels.map(h => (
          <option key={h.hotel_id} value={h.hotel_id}>
            {h.hotel_name} — {h.area}
          </option>
        ))}
      </select>

      {loading && <p>Loading...</p>}
      {profile && !loading && <TrustCard profile={profile} />}
    </div>
  )
}

export default App