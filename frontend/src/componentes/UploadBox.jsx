import './UploadBox.css'
import { useRef } from 'react'

export default function UploadBox({ image, setImage }) {
  const fileInput = useRef()

  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (ev) => setImage(ev.target.result)
      reader.readAsDataURL(file)
    }
  }

  const handleClick = () => {
    fileInput.current.click()
  }

  const handleReset = () => {
    setImage(null)
    if (fileInput.current) fileInput.current.value = ''
  }

  return (
    <div className="upload-box">
      {image ? (
        <>
          <img src={image} alt="preview" className="upload-preview" />
          <button className="upload-btn" type="button" onClick={handleReset} style={{marginTop: '1rem'}}>Recargar</button>
        </>
      ) : (
        <>
          <label className="upload-label" onClick={handleClick} style={{cursor: 'pointer'}}>
            Selecciona un archivo
          </label>
          <input
            ref={fileInput}
            id="file-upload"
            type="file"
            className="upload-input-hidden"
            onChange={handleChange}
            accept="image/*"
            style={{display: 'none'}}
          />
          <button className="upload-btn" type="button" onClick={handleClick}>Examinar...</button>
        </>
      )}
    </div>
  )
}
