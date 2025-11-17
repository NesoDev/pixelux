import './ProcessBox.css'
import { useState, useEffect } from 'react'

export default function ProcessBox({ imageLoaded, backendData }) {
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  useEffect(() => {
    setLoading(false)
    setDone(false)
  }, [imageLoaded])

  const handleProcess = () => {
    setLoading(true)
    setDone(false)
    setTimeout(() => {
      setLoading(false)
      setDone(true)
    }, 2200)
  }

  const handleDownload = () => {
    alert('Descargando archivo procesado...')
  }

  if (!imageLoaded) return null

  return (
    <div className="process-box">
      {!loading && !done && (
        <button className="process-btn" onClick={handleProcess}>
          Procesar
        </button>
      )}
      {loading && (
        <div className="loading-grid">
          {[...Array(9)].map((_, i) => (
            <span key={i} className={`loading-dot loading-dot-${i}`}></span>
          ))}
        </div>
      )}
      {!loading && done && (
        <button className="process-btn" onClick={handleDownload}>Descargar</button>
      )}
    </div>
  )
}
