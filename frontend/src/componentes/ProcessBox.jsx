import './ProcessBox.css'
import { useState, useEffect } from 'react'
import { processImage } from '../services/api'

export default function ProcessBox({ images, processingSettings, results, setResults, isProcessing, setIsProcessing }) {
  const [error, setError] = useState(null)
  const [processingProgress, setProcessingProgress] = useState({})

  useEffect(() => {
    // Reset when images change
    setError(null)
    setProcessingProgress({})
  }, [images])

  const handleProcessAll = async () => {
    if (images.length === 0) return

    setIsProcessing(true)
    setError(null)
    setResults([])
    setProcessingProgress({})

    const startTime = Date.now()

    try {
      // MOCK MODE - Simula procesamiento con imágenes reales del backend
      const MOCK_MODE = false

      if (MOCK_MODE) {
        // Simular carga de imágenes desde backend/shared/pixelart
        const mockBackendImages = [
          { original: '/backend/shared/pixelart/entradas/input-1.jpg', processed: '/backend/shared/pixelart/salidas/input-1_pixelart.png' },
          { original: '/backend/shared/pixelart/entradas/input-2.jpg', processed: '/backend/shared/pixelart/salidas/input-2_pixelart.png' },
          { original: '/backend/shared/pixelart/entradas/input-3.jpg', processed: '/backend/shared/pixelart/salidas/input-3_pixelart.png' }
        ]

        const promises = images.map(async (img, index) => {
          setProcessingProgress(prev => ({
            ...prev,
            [img.id]: { status: 'processing', progress: 0 }
          }))

          // Simular delay de procesamiento
          await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000))

          const imageStartTime = Date.now()

          // Usar imagen mock del backend (ciclar si hay más imágenes cargadas que mocks)
          const mockIndex = index % mockBackendImages.length

          // Simular resultado exitoso
          const imageEndTime = Date.now()
          const processingTime = imageEndTime - imageStartTime + 200 + Math.random() * 300

          setProcessingProgress(prev => ({
            ...prev,
            [img.id]: { status: 'completed', progress: 100 }
          }))

          return {
            id: img.id,
            originalName: img.name,
            originalData: img.data, // Usar la imagen que el usuario subió como "original"
            processedData: img.data, // Por ahora usar la misma (en producción sería la procesada)
            processingTime: Math.round(processingTime),
            metadata: {}
          }
        })

        const processedResults = await Promise.all(promises)
        const totalTime = Date.now() - startTime

        setResults(processedResults.map(r => ({
          ...r,
          totalBatchTime: totalTime
        })))

      } else {
        // Modo real - procesamiento con backend
        // Process all images in parallel
        const promises = images.map(async (img, index) => {
          setProcessingProgress(prev => ({
            ...prev,
            [img.id]: { status: 'processing', progress: 0 }
          }))

          try {
            const imageStartTime = Date.now()

            const result = await processImage({
              image: img.data,
              ...processingSettings
            })

            const imageEndTime = Date.now()
            const processingTime = imageEndTime - imageStartTime

            setProcessingProgress(prev => ({
              ...prev,
              [img.id]: { status: 'completed', progress: 100 }
            }))

            if (result.success) {
              return {
                id: img.id,
                originalName: img.name,
                originalData: img.data,
                processedData: result.image,
                processingTime: processingTime,
                metadata: result.metadata
              }
            } else {
              throw new Error(result.message || 'Processing failed')
            }
          } catch (err) {
            setProcessingProgress(prev => ({
              ...prev,
              [img.id]: { status: 'error', progress: 0 }
            }))
            console.error(`Error processing ${img.name}:`, err)
            return {
              id: img.id,
              originalName: img.name,
              error: err.message
            }
          }
        })

        const processedResults = await Promise.all(promises)
        const totalTime = Date.now() - startTime

        // Filter out errors and add total time info
        const successfulResults = processedResults.filter(r => !r.error)

        setResults(successfulResults.map(r => ({
          ...r,
          totalBatchTime: totalTime
        })))

        // Show error if some failed
        const failedCount = processedResults.filter(r => r.error).length
        if (failedCount > 0) {
          setError(`${failedCount} de ${images.length} imágenes fallaron al procesarse`)
        }
      }

    } catch (err) {
      console.error('Batch processing error:', err)
      setError(err.message || 'Error al procesar imágenes')
    } finally {
      setIsProcessing(false)
    }
  }

  if (images.length === 0) return null

  const hasResults = results.length > 0
  const processingList = Object.entries(processingProgress)

  return (
    <div className="process-box">
      {error && (
        <div className="error-message">
          <span className="status-icon error-icon">!</span> {error}
        </div>
      )}

      {!isProcessing && !hasResults && (
        <button className="process-btn process-btn-primary" onClick={handleProcessAll}>
          Procesar Todas ({images.length})
        </button>
      )}

      {isProcessing && (
        <div className="processing-status">
          <div className="processing-header">
            <div className="loading-spinner"></div>
            <span>Procesando en paralelo...</span>
          </div>
          <div className="progress-list">
            {images.map(img => {
              const progress = processingProgress[img.id]
              const status = progress?.status || 'waiting'

              return (
                <div key={img.id} className={`progress-item progress-${status}`}>
                  <span className="progress-name">{img.name}</span>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar"
                      style={{ width: status === 'processing' ? '50%' : status === 'completed' ? '100%' : '0%' }}
                    ></div>
                  </div>
                  <span className={`progress-status status-${status}`}>
                    {status === 'waiting' && <span className="status-icon">○</span>}
                    {status === 'processing' && <span className="status-icon processing">◐</span>}
                    {status === 'completed' && <span className="status-icon">✓</span>}
                    {status === 'error' && <span className="status-icon">✗</span>}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {hasResults && !isProcessing && (
        <div className="process-complete">
          <div className="success-message">
            <span className="status-icon success-icon">✓</span> {results.length} imagen{results.length !== 1 ? 'es' : ''} procesada{results.length !== 1 ? 's' : ''} exitosamente
          </div>
          <button className="process-btn process-btn-secondary" onClick={handleProcessAll}>
            Procesar Nuevamente
          </button>
        </div>
      )}
    </div>
  )
}

