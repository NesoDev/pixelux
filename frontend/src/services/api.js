/**
 * API service for communicating with Pixelux backend
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Process an image with the backend
 * @param {Object} data - Processing parameters
 * @param {string} data.image - Base64 encoded image
 * @param {string} data.algorithm - Processing algorithm ('no-dithering', 'dithering-floyd', or 'dithering-order')
 * @param {number} data.scale - Pixel size (1-20)
 * @param {string} data.palette - Color palette
 * @returns {Promise<Object>} Processing result
 */
export async function processImage(data) {
    try {
        const response = await fetch(`${API_URL}/api/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
        }

        const result = await response.json()
        return result
    } catch (error) {
        console.error('API Error:', error)
        throw error
    }
}

/**
 * Check API health status
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/api/health`)

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        return await response.json()
    } catch (error) {
        console.error('Health check failed:', error)
        throw error
    }
}
