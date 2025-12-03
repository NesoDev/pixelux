import './PerformanceMetrics.css'
import { useMemo } from 'react'

export default function PerformanceMetrics({ results }) {
    const metrics = useMemo(() => {
        if (results.length === 0) return null

        const totalProcessingTime = results.reduce((sum, r) => sum + r.processingTime, 0)
        const avgProcessingTime = totalProcessingTime / results.length
        const parallelTime = results[0]?.totalBatchTime || 0

        // Simulated serial time (sum of all individual times)
        const serialTime = totalProcessingTime
        const speedup = serialTime / parallelTime
        const efficiency = (speedup / results.length) * 100

        return {
            totalImages: results.length,
            avgProcessingTime: avgProcessingTime.toFixed(0),
            parallelTime: parallelTime.toFixed(0),
            serialTime: serialTime.toFixed(0),
            speedup: speedup.toFixed(2),
            efficiency: efficiency.toFixed(1),
            timeSaved: (serialTime - parallelTime).toFixed(0)
        }
    }, [results])

    if (!metrics) return null

    return (
        <div className="performance-metrics">
            <h2>Métricas de Rendimiento</h2>

            <div className="metrics-grid">
                <div className="metric-card metric-highlight">
                    <div className="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                            <polyline points="17 6 23 6 23 12" />
                        </svg>
                    </div>
                    <div className="metric-value">{metrics.speedup}x</div>
                    <div className="metric-label">Aceleración (Speedup)</div>
                </div>

                <div className="metric-card">
                    <div className="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <polyline points="12 6 12 12 16 14" />
                        </svg>
                    </div>
                    <div className="metric-value">{metrics.efficiency}%</div>
                    <div className="metric-label">Eficiencia</div>
                </div>

                <div className="metric-card">
                    <div className="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="12" y1="2" x2="12" y2="6" />
                            <line x1="12" y1="18" x2="12" y2="22" />
                            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
                            <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
                            <line x1="2" y1="12" x2="6" y2="12" />
                            <line x1="18" y1="12" x2="22" y2="12" />
                            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
                            <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
                        </svg>
                    </div>
                    <div className="metric-value">{metrics.timeSaved}ms</div>
                    <div className="metric-label">Tiempo Ahorrado</div>
                </div>
            </div>

            <div className="comparison-chart">
                <div className="chart-row">
                    <div className="chart-label">Proceso Serial (simulado)</div>
                    <div className="chart-bar-container">
                        <div
                            className="chart-bar chart-bar-serial"
                            style={{
                                width: `${(parseFloat(metrics.serialTime) / Math.max(parseFloat(metrics.serialTime), parseFloat(metrics.parallelTime))) * 100}%`
                            }}
                        >
                            <span className="chart-value">{metrics.serialTime}ms</span>
                        </div>
                    </div>
                </div>

                <div className="chart-row">
                    <div className="chart-label">Proceso Paralelo (real)</div>
                    <div className="chart-bar-container">
                        <div
                            className="chart-bar chart-bar-parallel"
                            style={{
                                width: `${(parseFloat(metrics.parallelTime) / Math.max(parseFloat(metrics.serialTime), parseFloat(metrics.parallelTime))) * 100}%`
                            }}
                        >
                            <span className="chart-value">{metrics.parallelTime}ms</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="metrics-details">
                <div className="detail-item">
                    <span className="detail-label">Imágenes procesadas:</span>
                    <span className="detail-value">{metrics.totalImages}</span>
                </div>
                <div className="detail-item">
                    <span className="detail-label">Tiempo promedio por imagen:</span>
                    <span className="detail-value">{metrics.avgProcessingTime}ms</span>
                </div>
            </div>

            <div className="metrics-explanation">
                <p>
                    <strong>Speedup:</strong> Indica cuántas veces más rápido es el procesamiento paralelo comparado con procesamiento serial.
                </p>
                <p>
                    <strong>Eficiencia:</strong> Mide qué tan bien se utilizan los recursos de procesamiento paralelo (GPU/CUDA).
                </p>
            </div>
        </div>
    )
}
