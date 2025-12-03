import './Menu.css'

export default function Menu({ dithering, setDithering, scale, setScale, palette, setPalette }) {
  return (
    <div className="menu-box">
      <div className="menu-grid">
        {/* Column 1: Algoritmo */}
        <div className="menu-column">
          <label className="menu-label">Algoritmo</label>
          <select
            className="menu-select"
            value={dithering ? 'dithering' : 'no-dithering'}
            onChange={e => setDithering(e.target.value === 'dithering')}
          >
            <option value="dithering">Dithering</option>
            <option value="no-dithering">No dithering</option>
          </select>
        </div>

        {/* Column 2: Escala */}
        <div className="menu-column">
          <label className="menu-label" htmlFor="scale-range">Escala ({scale}x)</label>
          <input
            id="scale-range"
            className="menu-range"
            type="range"
            min="1"
            max="20"
            value={scale}
            onChange={e => setScale(Number(e.target.value))}
          />
        </div>

        {/* Column 3: Paleta */}
        <div className="menu-column">
          <label className="menu-label" htmlFor="palette-select">Paleta de colores</label>
          <select
            id="palette-select"
            className="menu-select"
            value={palette}
            onChange={e => setPalette(e.target.value)}
          >
            <option value="free">Free colors</option>
            <option value="paid">Paid colors</option>
            <option value="both">Both</option>
          </select>
        </div>
      </div>
    </div>
  )
}

