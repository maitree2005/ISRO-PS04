"use client"
import React, { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup, ImageOverlay } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

function ControlPanel({ result, onClose }) {
  if (!result) return null
  return (
    <div style={{ position: 'absolute', right: 12, top: 12, zIndex: 1000, background: 'white', padding: 12, borderRadius: 8, boxShadow: '0 2px 6px rgba(0,0,0,0.2)' }}>
      <strong>Simulation Result</strong>
      <div>Node: {result.node_removed}</div>
      <div>Resilience Index: {result.resilience_index}</div>
      <div>Baseline LCC: {result.baseline_lcc_size}</div>
      <button style={{ marginTop: 8 }} onClick={onClose}>Close</button>
    </div>
  )
}

export default function MapComponent() {
  const [geojson, setGeojson] = useState(null)
  const [nodes, setNodes] = useState([])
  const [simResult, setSimResult] = useState(null)
  const [showHeat, setShowHeat] = useState(false)
  const [modelList, setModelList] = useState([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('')
  const [preloadedHandles, setPreloadedHandles] = useState([])
  const [activeHandle, setActiveHandle] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [maskUrl, setMaskUrl] = useState(null)
  const fileInputRef = useRef(null)
  //const [mapInstance, setMapInstance] = useState(null)

  useEffect(() => {
    // Fetch graph GeoJSON
    fetch('http://127.0.0.1:8000/api/graph')
      .then((r) => r.json())
      .then((data) => setGeojson(data))
      .catch((err) => console.error(err))

    // Fetch criticality scores
    fetch('http://127.0.0.1:8000/api/criticality')
      .then((r) => r.json())
      .then((data) => {
        const list = data.top_bottlenecks || []
        setNodes(list)
      })
      .catch(() => {})

    // Fetch available model checkpoints and preloaded status
    fetch('http://127.0.0.1:8000/api/models/list')
      .then((r) => r.json())
      .then((data) => setModelList(data || []))
      .catch(() => setModelList([]))

    fetch('http://127.0.0.1:8000/api/models/status')
      .then((r) => r.json())
      .then((data) => setPreloadedHandles(data.preloaded || []))
      .catch(() => setPreloadedHandles([]))
  }, [])

  const onNodeClick = async (node: any) => {
  try {
    const resp = await fetch("http://127.0.0.1:8000/api/simulate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        node_id: node.node_id,
      }),
    });

    const data = await resp.json();
    setSimResult(data);
  } catch (e) {
    console.error(e);
  }
};
  const colorForScore = (s) => {
    // s expected 0..1
    if (s === null || s === undefined) return '#00ff00'
    const r = Math.round(Math.min(1, s) * 255)
    const g = Math.round((1 - Math.min(1, s)) * 180)
    return `rgb(${r},${g},0)`
  }

  const exportGeoJSON = async () => {
  try {
    const resp = await fetch("http://127.0.0.1:8000/api/graph");

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();

    const blob = new Blob(
      [JSON.stringify(data, null, 2)],
      { type: "application/geo+json" }
    );

    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "graph.geojson";

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    window.URL.revokeObjectURL(url);

    console.log("✅ GeoJSON exported successfully.");
  } catch (err) {
    console.error("GeoJSON Export Error:", err);
    alert("Failed to export GeoJSON. Make sure the FastAPI backend is running.");
  }
};

  const exportCSV = async () => {
    try {
      const resp = await fetch('/api/criticality')
      const data = await resp.json()
      const rows = ['node_id,lat,lon,betweenness_score,criticality_rank,label']
      for (const n of data.top_bottlenecks || []) {
        rows.push([n.node_id, n.lat, n.lon, n.betweenness_score, n.criticality_rank, (n.label||'')].join(','))
      }
      const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'criticality.csv'
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error(e)
    }
  }

  const preloadModel = async () => {
    try {
      if (!selectedCheckpoint) return
      const form = new FormData()
      form.append('checkpoint', selectedCheckpoint)
      const resp = await fetch('/api/models/preload', { method: 'POST', body: form })
      const data = await resp.json()
      if (resp.ok) {
        setPreloadedHandles((h) => Array.from(new Set([...h, data.handle])))
      } else {
        console.error('Preload failed', data)
      }
    } catch (e) {
      console.error(e)
    }
  }

  const unloadModel = async (handle) => {
    try {
      const form = new FormData()
      form.append('handle', handle)
      const resp = await fetch('/api/models/unload', { method: 'POST', body: form })
      if (resp.ok) {
        setPreloadedHandles((h) => h.filter((x) => x !== handle))
        if (activeHandle === handle) setActiveHandle(null)
      } else {
        console.error('Unload failed', await resp.text())
      }
    } catch (e) {
      console.error(e)
    }
  }

  const refreshModels = async () => {
    try {
      const [listResp, statusResp] = await Promise.all([fetch('/api/models/list'), fetch('/api/models/status')])
      const list = await listResp.json()
      const status = await statusResp.json()
      setModelList(list || [])
      setPreloadedHandles(status.preloaded || [])
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div style={{ height: '100vh', width: '100%', position: 'relative' }}>
      <MapContainer
  center={[12.9716, 77.5946]}
  zoom={12}
  style={{ height: '100%', width: '100%' }}
>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='© OpenStreetMap contributors'
        />

        {geojson && (
          <GeoJSON data={geojson} style={() => ({ color: '#ffffff', weight: 2 })} />
        )}

        {nodes.map((n) => (
          <CircleMarker
            key={n.node_id}
            center={[n.lat, n.lon]}
            radius={showHeat ? Math.max(8, (n.betweenness_score || 0) * 40) : 8}
            pathOptions={{ color: colorForScore(n.betweenness_score), fillOpacity: showHeat ? 0.25 : 0.8 }}
            eventHandlers={{ click: () => onNodeClick(n) }}
          >
            <Popup>
              <div>
                <strong>{n.label || n.node_id}</strong>
                <div>Score: {n.betweenness_score}</div>
                <button onClick={() => onNodeClick(n)}>Simulate removal</button>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>

      <ControlPanel result={simResult} onClose={() => setSimResult(null)} />

          <div style={{ position: 'absolute', left: 12, top: 12, zIndex: 1000, background: 'white', padding: 8, borderRadius: 8 }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => setShowHeat((s) => !s)}>{showHeat ? 'Hide Heat' : 'Show Heat'}</button>
                <button onClick={exportGeoJSON}>Export GeoJSON</button>
                <button onClick={exportCSV}>Export CSV</button>
              </div>

              <div style={{ marginTop: 8 }}>
                <div style={{ marginBottom: 6 }}><strong>Model Management</strong></div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <select value={selectedCheckpoint} onChange={(e) => setSelectedCheckpoint(e.target.value)}>
                    <option value="">Select checkpoint...</option>
                    {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <button onClick={preloadModel}>Preload</button>
                  <button onClick={refreshModels}>Refresh</button>
                </div>
                <div style={{ marginTop: 6 }}>
                  <div>Preloaded: {preloadedHandles.length === 0 ? 'none' : preloadedHandles.join(', ')}</div>
                  <div style={{ marginTop: 6 }}>
                    <select value={activeHandle || ''} onChange={(e) => setActiveHandle(e.target.value)}>
                      <option value="">Active handle (none)</option>
                      {preloadedHandles.map((h) => <option key={h} value={h}>{h}</option>)}
                    </select>
                    <button disabled={!activeHandle} onClick={() => unloadModel(activeHandle)} style={{ marginLeft: 8 }}>Unload active</button>
                  </div>
                </div>
                <div style={{ marginTop: 8 }}>
                  <input ref={fileInputRef} type="file" accept="image/*" />
                  <button onClick={async () => {
                    const f = fileInputRef.current && fileInputRef.current.files && fileInputRef.current.files[0]
                    if (!f) return alert('Choose an image first')
                    setUploading(true)
                    try {
                      const fd = new FormData()
                      fd.append('image', f)
                      if (activeHandle) fd.append('handle', activeHandle)
                      const resp = await fetch('/api/predict', { method: 'POST', body: fd })
                      if (!resp.ok) {
                        const txt = await resp.text()
                        throw new Error(txt || 'Predict failed')
                      }
                      const blob = await resp.blob()
                      const url = URL.createObjectURL(blob)
                      setMaskUrl(url)
                    } catch (err) {
                      console.error(err)
                      alert('Prediction failed: ' + err.message)
                    } finally {
                      setUploading(false)
                    }
                  }} disabled={uploading} style={{ marginLeft: 8 }}>{uploading ? 'Predicting...' : 'Predict'}</button>
                  {maskUrl && (
                    <>
                      <a href={maskUrl} target="_blank" rel="noreferrer" style={{ marginLeft: 8 }}>Open mask</a>
                      <button onClick={() => { URL.revokeObjectURL(maskUrl); setMaskUrl(null) }} style={{ marginLeft: 8 }}>Clear overlay</button>
                    </>
                  )}
                </div>
              </div>
          </div>
    </div>
  )
}
