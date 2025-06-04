//! Core MMX format implementation

use super::{MMXError, MMXHeader, ChunkDirectory, CompressionType};
use std::io::{Read, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// MMX file format constants
pub const MMX_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 4096; // Fixed header size for alignment

/// Write MMX file header
pub fn write_header<W: Write + Seek>(
    writer: &mut W,
    header: &MMXHeader
) -> Result<(), MMXError> {
    // Magic bytes
    writer.write_all(crate::MAGIC_BYTES)?;
    
    // Version
    writer.write_u32::<LittleEndian>(header.version)?;
    
    // Serialize header as JSON (for flexibility)
    let header_json = serde_json::to_vec(header)
        .map_err(|e| MMXError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e
        )))?;
    
    // Write header size and data
    writer.write_u32::<LittleEndian>(header_json.len() as u32)?;
    writer.write_all(&header_json)?;
    
    // Pad to fixed header size
    let current_pos = writer.stream_position()?;
    let padding = HEADER_SIZE as u64 - current_pos;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding as usize])?;
    }
    
    Ok(())
}

/// Read MMX file header
pub fn read_header<R: Read + Seek>(reader: &mut R) -> Result<MMXHeader, MMXError> {
    // Verify magic bytes
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != crate::MAGIC_BYTES {
        return Err(MMXError::InvalidMagic);
    }
    
    // Read version
    let version = reader.read_u32::<LittleEndian>()?;
    if version != MMX_VERSION {
        return Err(MMXError::UnsupportedVersion(version));
    }
    
    // Read header JSON
    let header_size = reader.read_u32::<LittleEndian>()?;
    let mut header_data = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_data)?;
    
    let header: MMXHeader = serde_json::from_slice(&header_data)
        .map_err(|e| MMXError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e
        )))?;
    
    // Seek past padding
    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    
    Ok(header)
}

/// Compress data using specified method
pub fn compress_data(data: &[u8], compression: CompressionType) -> Result<Vec<u8>, MMXError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Lz4 => {
            Ok(lz4_flex::compress_prepend_size(data))
        },
        CompressionType::Zlib => {
            use flate2::write::ZlibEncoder;
            use flate2::Compression;
            
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(data)
                .map_err(|e| MMXError::Compression(e.to_string()))?;
            encoder.finish()
                .map_err(|e| MMXError::Compression(e.to_string()))
        }
    }
}

/// Decompress data using specified method
pub fn decompress_data(data: &[u8], compression: CompressionType) -> Result<Vec<u8>, MMXError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Lz4 => {
            lz4_flex::decompress_size_prepended(data)
                .map_err(|e| MMXError::Compression(e.to_string()))
        },
        CompressionType::Zlib => {
            use flate2::read::ZlibDecoder;
            
            let mut decoder = ZlibDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| MMXError::Compression(e.to_string()))?;
            Ok(decompressed)
        }
    }
}

/// Calculate SHA-256 checksum
pub fn calculate_checksum(data: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}
