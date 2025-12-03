import './ImageComparison.css'
import { useState, useRef, useEffect } from 'react'

export default function ImageComparison({ originalImage, processedImage, originalName }) {
    const [sliderPosition, setSliderPosition] = useState(50)
    const containerRef = useRef(null)
    const [isDragging, setIsDragging] = useState(false)

    const handleMouseDown = () => setIsDragging(true)

    useEffect(() => {
        const handleMouseUp = () => setIsDragging(false)
        const handleMouseMove = (e) => {
            if (isDragging && containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect()
                const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width))
                const percentage = (x / rect.width) * 100
                setSliderPosition(percentage)
            }
        }

        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove)
            document.addEventListener('mouseup', handleMouseUp)
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove)
            document.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isDragging])

    const handleTouchMove = (e) => {
        if (containerRef.current) {
            const touch = e.touches[0]
            const rect = containerRef.current.getBoundingClientRect()
            const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width))
            const percentage = (x / rect.width) * 100
            setSliderPosition(percentage)
        }
    }

    return (
        <div
            ref={containerRef}
            className="image-comparison"
            onTouchMove={handleTouchMove}
        >
            <div className="comparison-container">
                {/* Processed Image (Background) */}
                <div
                    className="image-layer processed-layer"
                    style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
                >
                    <img src={processedImage} alt={`${originalName} - Procesada`} />
                    <div className="image-label image-label-right">Procesada</div>
                </div>

                {/* Original Image (Foreground with clip) */}
                <div
                    className="image-layer original-layer"
                    style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
                >
                    <img src={originalImage} alt={`${originalName} - Original`} />
                    <div className="image-label image-label-left">Original</div>
                </div>

                {/* Slider */}
                <div
                    className="slider-line"
                    style={{ left: `${sliderPosition}%` }}
                    onMouseDown={handleMouseDown}
                >
                    <div className="slider-handle">
                        <div className="slider-arrows">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="15 18 9 12 15 6" />
                            </svg>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="9 18 15 12 9 6" />
                            </svg>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
