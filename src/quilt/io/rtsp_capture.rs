pub struct RtspCapture {
    cap: opencv::videoio::VideoCapture,
    pub url: String,
}
impl RtspCapture {
    pub fn new(url: &str) -> anyhow::Result<Self> {
        use opencv::videoio::{VideoCapture, CAP_FFMPEG, CAP_PROP_BUFFERSIZE};
        let mut cap = VideoCapture::from_file(url, CAP_FFMPEG)?;
        cap.set(CAP_PROP_BUFFERSIZE, 2.0)?;
        Ok(Self {
            cap,
            url: url.into(),
        })
    }
    pub fn read(&mut self) -> anyhow::Result<Option<(opencv::core::Mat, std::time::SystemTime)>> {
        let mut frame = opencv::core::Mat::default();
        if !self.cap.read(&mut frame)? || frame.empty() {
            return Ok(None);
        }
        Ok(Some((frame, std::time::SystemTime::now())))
    }
}
