//! MMX: Multimedia Matrix Format
//!
//! A unified tensor-backed multimedia format for efficient storage and streaming
//! of high-dimensional data across all modalities.

pub mod format;
pub mod chunks;
pub mod api;

pub use format::*;
pub use chunks::*;
pub use api::*;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// MMX file format errors
#[derive(thiserror::Error, Debug)]
pub enum MMXError {
    #[error("Invalid magic bytes")]
    InvalidMagic,
    
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Chunk not found: {0}")]
    ChunkNotFound(String),
    
    #[error("Invalid chunk type: {0}")]
    InvalidChunkType(u8),
    
    #[error("Compression error: {0}")]
    Compression(String),
    
    #[error("Data integrity error: {0}")]
    DataIntegrity(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Supported chunk types in MMX format
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkType {
    Tensor = 0x01,
    Sequence = 0x02,
    Text = 0x03,
    Mesh = 0x04,
    Embedding = 0x05,
    Meta = 0x06,
}

impl TryFrom<u8> for ChunkType {
    type Error = MMXError;
    
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(ChunkType::Tensor),
            0x02 => Ok(ChunkType::Sequence),
            0x03 => Ok(ChunkType::Text),
            0x04 => Ok(ChunkType::Mesh),
            0x05 => Ok(ChunkType::Embedding),
            0x06 => Ok(ChunkType::Meta),
            _ => Err(MMXError::InvalidChunkType(value)),
        }
    }
}

/// Compression methods supported by MMX
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Lz4,
    Zlib,
}

/// Global metadata for MMX files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMXHeader {
    pub version: u32,
    pub file_id: Uuid,
    pub created: DateTime<Utc>,
    pub creator: String,
    pub model_uuid: Option<Uuid>,
    pub modality_summary: HashMap<String, usize>,
    pub total_chunks: u64,
    pub file_size: u64,
}

/// Directory entry for chunk lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDirectory {
    pub chunks: HashMap<String, ChunkEntry>,
}

/// Individual chunk metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub name: String,
    pub chunk_type: ChunkType,
    pub offset: u64,
    pub size: u64,
    pub compression: CompressionType,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub logical_region: Option<String>,
    pub checksum: Option<[u8; 32]>, // SHA-256
}
