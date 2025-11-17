import './App.css'
import Header from './componentes/Header'
import UploadBox from './componentes/UploadBox'
import Menu from './componentes/Menu'
import ProcessBox from './componentes/ProcessBox'
import { useState, useMemo } from 'react'

function App() {
  const [image, setImage] = useState(null)
  const [dithering, setDithering] = useState(false)
  const [scale, setScale] = useState(5)
  const [palette, setPalette] = useState('free')

  // Objeto listo para enviar al backend
  const backendData = useMemo(() => ({
    image, // base64 o null
    algorithm: dithering ? 'dithering' : 'no-dithering',
    scale,
    palette
  }), [image, dithering, scale, palette])

  return (
    <div className="app">
      <Header />
      <UploadBox image={image} setImage={setImage} />
      <Menu dithering={dithering} setDithering={setDithering} scale={scale} setScale={setScale} palette={palette} setPalette={setPalette} />
      <ProcessBox imageLoaded={!!image} backendData={backendData} />
    </div>
  )
}

export default App
