import './ResultsGallery.css'
import { useState } from 'react'
import ImageComparison from './ImageComparison'

export default function ResultsGallery({ results }) {
    const [selectedResult, setSelectedResult] = useState(null)
    const [downloadingAll, setDownloadingAll] = useState(false)

    const handleDownload = (result) => {
        try {
            const link = document.createElement('a')
            link.href = result.processedData
            link.download = `pixelart_${result.originalName}`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
        } catch (err) {
            console.error('Download error:', err)
        }
    }

    const handleDownloadAll = async () => {
        setDownloadingAll(true)

        // Simple sequential download (for better browser compatibility)
        for (const result of results) {
            await new Promise(resolve => {
                handleDownload(result)
                setTimeout(resolve, 500) // Small delay between downloads
            })
        }

        setDownloadingAll(false)
    }

    if (results.length === 0) return null

    return (
        <>
            <div className="results-gallery">
                <div className="gallery-header">
                    <h2>Resultados ({results.length})</h2>
                    <button
                        className="download-all-btn"
                        onClick={handleDownloadAll}
                        disabled={downloadingAll}
                    >
                        {downloadingAll ? 'Descargando...' : 'Descargar Todas'}
                    </button>
                </div>

                <div className="gallery-grid">
                    {results.map((result) => (
                        <div
                            key={result.id}
                            className="gallery-item"
                            onClick={() => setSelectedResult(result)}
                        >
                            <div className="gallery-item-preview">
                                <img src={result.processedData} alt={result.originalName} />
                            </div>
                            <div className="gallery-item-info">
                                <span className="gallery-item-name">{result.originalName}</span>
                                <span className="gallery-item-time">{result.processingTime}ms</span>
                            </div>
                            <button
                                className="gallery-item-download"
                                onClick={(e) => {
                                    e.stopPropagation()
                                    handleDownload(result)
                                }}
                                title="Descargar"
                            >
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
                                </svg>
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Modal for detailed comparison */}
            {selectedResult && (
                <div className="comparison-modal" onClick={() => setSelectedResult(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="modal-close" onClick={() => setSelectedResult(null)}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                <line x1="18" y1="6" x2="6" y2="18" />
                                <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                        </button>
                        <h3>{selectedResult.originalName}</h3>
                        <ImageComparison
                            originalImage={selectedResult.originalData}
                            processedImage={selectedResult.processedData}
                            originalName={selectedResult.originalName}
                        />
                        <div className="modal-actions">
                            <button
                                className="modal-btn modal-btn-primary"
                                onClick={() => handleDownload(selectedResult)}
                            >
                                Descargar
                            </button>
                            <button
                                className="modal-btn modal-btn-secondary"
                                onClick={() => setSelectedResult(null)}
                            >
                                Cerrar
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}
