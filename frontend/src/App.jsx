import './App.css'
import Header from './componentes/Header'
import UploadBox from './componentes/UploadBox'
import Menu from './componentes/Menu'
import ProcessBox from './componentes/ProcessBox'
import ResultsGallery from './componentes/ResultsGallery'
import PerformanceMetrics from './componentes/PerformanceMetrics'
import { useState, useMemo } from 'react'

function App() {
  // Multiple images support
  const [images, setImages] = useState([])
  const [dithering, setDithering] = useState(false)
  const [scale, setScale] = useState(5)
  const [palette, setPalette] = useState('free')

  // Processing results: array of {original, processed, time, metadata}
  const [results, setResults] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)

  // MOCK DATA - Set to true to preview components (moved to ProcessBox)
  const USE_MOCK_DATA = false

  const mockResults = USE_MOCK_DATA ? [
    {
      id: 1,
      originalName: "sample1.jpg",
      originalData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23555' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EOriginal%3C/text%3E%3C/svg%3E",
      processedData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23777' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EProcessed%3C/text%3E%3C/svg%3E",
      processingTime: 234,
      totalBatchTime: 567,
      metadata: {}
    },
    {
      id: 2,
      originalName: "sample2.png",
      originalData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23666' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EOriginal%3C/text%3E%3C/svg%3E",
      processedData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23888' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EProcessed%3C/text%3E%3C/svg%3E",
      processingTime: 189,
      totalBatchTime: 567,
      metadata: {}
    },
    {
      id: 3,
      originalName: "sample3.jpg",
      originalData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23444' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EOriginal%3C/text%3E%3C/svg%3E",
      processedData: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23999' width='200' height='200'/%3E%3Ctext x='50%25' y='50%25' fill='white' text-anchor='middle' dy='.3em' font-size='16'%3EProcessed%3C/text%3E%3C/svg%3E",
      processingTime: 144,
      totalBatchTime: 567,
      metadata: {}
    }
  ] : []

  // Processing settings for all images
  const processingSettings = useMemo(() => ({
    algorithm: dithering ? 'dithering' : 'no-dithering',
    scale,
    palette
  }), [dithering, scale, palette])

  return (
    <div className="app">
      <Header />
      <Menu
        dithering={dithering}
        setDithering={setDithering}
        scale={scale}
        setScale={setScale}
        palette={palette}
        setPalette={setPalette}
      />
      <UploadBox images={images} setImages={setImages} />
      <ProcessBox
        images={images}
        processingSettings={processingSettings}
        results={results}
        setResults={setResults}
        isProcessing={isProcessing}
        setIsProcessing={setIsProcessing}
      />
      {(results.length > 0 || mockResults.length > 0) && (
        <>
          <PerformanceMetrics results={mockResults.length > 0 ? mockResults : results} />
          <ResultsGallery results={mockResults.length > 0 ? mockResults : results} />
        </>
      )}
    </div>
  )
}

export default App
