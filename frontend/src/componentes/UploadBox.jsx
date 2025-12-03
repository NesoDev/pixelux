import './UploadBox.css'
import { useRef, useState } from 'react'

export default function UploadBox({ images, setImages }) {
  const fileInput = useRef()
  const [dragActive, setDragActive] = useState(false)

  const processFiles = (files) => {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
    const imageFiles = Array.from(files).filter(file =>
      file.type.startsWith('image/') && allowedTypes.includes(file.type.toLowerCase())
    )

    const rejected = files.length - imageFiles.length
    if (rejected > 0) {
      alert(`${rejected} archivo(s) rechazado(s). Formatos permitidos: PNG, JPG, JPEG, WebP`)
    }

    imageFiles.forEach(file => {
      const reader = new FileReader()
      reader.onload = (e) => {
        setImages(prev => [...prev, {
          id: Date.now() + Math.random(),
          name: file.name,
          data: e.target.result,
          size: file.size
        }])
      }
      reader.readAsDataURL(file)
    })
  }

  const handleChange = (e) => {
    processFiles(e.target.files)
    e.target.value = '' // Reset input to allow same file again
  }

  const handleClick = () => {
    fileInput.current.click()
  }

  const handleRemove = (id) => {
    setImages(prev => prev.filter(img => img.id !== id))
  }

  const handleReset = () => {
    setImages([])
    if (fileInput.current) fileInput.current.value = ''
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files)
    }
  }

  return (
    <div className="upload-box">
      {images.length > 0 ? (
        <>
          <div className="upload-header">
            <span className="image-counter">
              {images.length} imagen{images.length !== 1 ? 'es' : ''} cargada{images.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="upload-preview-grid">
            {images.map(img => (
              <div key={img.id} className="preview-item">
                <img src={img.data} alt={img.name} className="preview-thumbnail" />
                <button
                  className="remove-btn"
                  onClick={() => handleRemove(img.id)}
                  title="Eliminar"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
                <span className="preview-name">{img.name}</span>
              </div>
            ))}
          </div>
          <div className="upload-actions">
            <button className="upload-btn" type="button" onClick={handleClick}>
              Agregar más
            </button>
            <button className="upload-btn upload-btn-secondary" type="button" onClick={handleReset}>
              Limpiar todo
            </button>
          </div>
        </>
      ) : (
        <div
          className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={handleClick}
        >
          <div className="upload-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
            </svg>
          </div>
          <label className="upload-label">
            Arrastra imágenes aquí o haz clic para seleccionar
          </label>
          <p className="upload-hint">PNG, JPG, JPEG, WebP</p>
        </div>
      )}
      {/* Hidden file input - always rendered */}
      <input
        ref={fileInput}
        type="file"
        className="upload-input-hidden"
        onChange={handleChange}
        accept="image/png,image/jpeg,image/jpg,image/webp"
        multiple
        style={{ display: 'none' }}
      />
    </div>
  )
}
