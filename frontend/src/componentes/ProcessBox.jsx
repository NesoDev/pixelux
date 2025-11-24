import './ProcessBox.css'
import { useState, useEffect } from 'react'
import { processImage } from '../services/api'

export default function ProcessBox({ imageLoaded, backendData }) {
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)
  const [processedImage, setProcessedImage] = useState(null)
  const [processingTime, setProcessingTime] = useState(null)

  useEffect(() => {
    setLoading(false)
    setDone(false)
    setError(null)
    setProcessedImage(null)
    setProcessingTime(null)
  }, [imageLoaded])

  const handleProcess = async () => {
    setLoading(true)
    setDone(false)
    setError(null)
    setProcessingTime(null)

    try {
      const result = await processImage(backendData)

      if (result.success) {
        setProcessedImage(result.image)
        setProcessingTime(result.processing_time_ms)
        setDone(true)
      } else {
        setError(result.message || 'Processing failed')
      }
    } catch (err) {
      console.error('Processing error:', err)
      setError(err.message || 'Failed to process image. Please check if the API server is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    if (!processedImage) return

    try {
      // Create a temporary link to download the image
      const link = document.createElement('a')
      link.href = processedImage
      link.download = `pixelart_${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (err) {
      console.error('Download error:', err)
      setError('Failed to download image')
    }
  }

  if (!imageLoaded) return null

  return (
    <div className="process-box">
      {error && (
        <div className="error-message" style={{
          color: '#ff4444',
          padding: '1rem',
          marginBottom: '1rem',
          borderRadius: '8px',
          backgroundColor: 'rgba(255, 68, 68, 0.1)',
          border: '1px solid rgba(255, 68, 68, 0.3)'
        }}>
          ⚠️ {error}
        </div>
      )}

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
        <>
          {processingTime && (
            <div style={{
              fontSize: '0.9rem',
              color: '#888',
              marginBottom: '0.5rem'
            }}>
              Procesado en {processingTime.toFixed(0)}ms
            </div>
          )}
          <button className="process-btn" onClick={handleDownload}>
            Descargar
          </button>
        </>
      )}
    </div>
  )
}
