import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('Frontend runtime error', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ minHeight: '100vh', padding: '32px', background: '#f6f1e7', color: '#1f221c', fontFamily: 'IBM Plex Sans, Segoe UI, sans-serif' }}>
          <div style={{ maxWidth: '900px', margin: '0 auto', padding: '24px', borderRadius: '24px', background: 'rgba(255,252,247,0.94)', border: '1px solid rgba(48,55,42,0.14)' }}>
            <p style={{ margin: 0, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#c65d2f', fontWeight: 700, fontSize: '0.8rem' }}>Frontend error boundary</p>
            <h1 style={{ margin: '12px 0', fontSize: '2rem' }}>UI bị lỗi runtime nên tôi chặn trắng màn hình.</h1>
            <p style={{ color: '#5f6359', lineHeight: 1.7 }}>Mở DevTools Console để xem lỗi chi tiết. Nếu anh gửi ảnh Console, tôi sẽ sửa đúng điểm gãy tiếp theo.</p>
            <pre style={{ overflow: 'auto', padding: '14px', borderRadius: '16px', background: '#fff', border: '1px solid rgba(48,55,42,0.12)' }}>{String(this.state.error || 'Unknown error')}</pre>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
)
