//! High-level API for working with MMX files

use super::{
    MMXError, MMXHeader, ChunkDirectory, ChunkEntry, ChunkType, CompressionType,
    TensorData, MeshData, EmbeddingData,
    format::*
};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::path::Path;
use std::collections::HashMap;
use uuid::Uuid;

/// Main MMX file interface
pub struct MMXFile {
    file: File,
    header: MMXHeader,
    directory: ChunkDirectory,
    mode: MMXMode,
}

#[derive(Debug, Clone, Copy)]
pub enum MMXMode {
    Read,
    Write,
    Append,
}

impl MMXFile {
    /// Create a new MMX file
    pub fn create<P: AsRef<Path>>(path: P, creator: String) -> Result<Self, MMXError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        
        let header = MMXHeader {
            version: crate::mmx::format::MMX_VERSION,
            file_id: Uuid::new_v4(),
            created: chrono::Utc::now(),
            creator,
            model_uuid: None,
            modality_summary: HashMap::new(),
            total_chunks: 0,
            file_size: 0,
        };
        
        let directory = ChunkDirectory {
            chunks: HashMap::new(),
        };
        
        let mut mmx_file = Self {
            file,
            header,
            directory,
            mode: MMXMode::Write,
        };
        
        mmx_file.write_header()?;
        Ok(mmx_file)
    }
    
    /// Open existing MMX file
    pub fn open<P: AsRef<Path>>(path: P, mode: MMXMode) -> Result<Self, MMXError> {
        let file = match mode {
            MMXMode::Read => OpenOptions::new().read(true).open(path)?,
            MMXMode::Write => OpenOptions::new().read(true).write(true).open(path)?,
            MMXMode::Append => OpenOptions::new().read(true).write(true).append(true).open(path)?,
        };
        
        let mut reader = BufReader::new(&file);
        let header = read_header(&mut reader)?;
        
        // Read directory (simplified - would be at end of file)
        let directory = ChunkDirectory {
            chunks: HashMap::new(), // Would load from file
        };
        
        Ok(Self {
            file,
            header,
            directory,
            mode,
        })
    }
    
    /// Write tensor chunk
    pub fn write_tensor(&mut self, path: &str, tensor: TensorData) -> Result<(), MMXError> {
        self.write_chunk(path, ChunkType::Tensor, tensor.to_bytes(), CompressionType::Lz4)
    }
    
    /// Read tensor chunk
    pub fn read_tensor(&mut self, path: &str) -> Result<TensorData, MMXError> {
        let data = self.read_chunk(path)?;
        TensorData::from_bytes(&data)
    }
    
    /// Write mesh chunk
    pub fn write_mesh(&mut self, path: &str, mesh: MeshData) -> Result<(), MMXError> {
        let data = bincode::serialize(&mesh)
            .map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        self.write_chunk(path, ChunkType::Mesh, data, CompressionType::Lz4)
    }
    
    /// Read mesh chunk
    pub fn read_mesh(&mut self, path: &str) -> Result<MeshData, MMXError> {
        let data = self.read_chunk(path)?;
        bincode::deserialize(&data)
            .map_err(|e| MMXError::DataIntegrity(e.to_string()))
    }
    
    /// Write embedding chunk
    pub fn write_embedding(&mut self, path: &str, embedding: EmbeddingData) -> Result<(), MMXError> {
        let data = bincode::serialize(&(
            embedding.matrix.shape().to_vec(),
            embedding.matrix.as_slice().unwrap().to_vec(),
            embedding.method,
            embedding.labels
        )).map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        
        self.write_chunk(path, ChunkType::Embedding, data, CompressionType::Lz4)
    }
    
    /// Create a group (directory structure)
    pub fn create_group(&mut self, _path: &str) -> Result<(), MMXError> {
        // Groups are implicit in the path structure
        Ok(())
    }
    
    /// List all chunks
    pub fn list_chunks(&self) -> Vec<String> {
        self.directory.chunks.keys().cloned().collect()
    }
    
    /// Get chunk metadata
    pub fn get_chunk_info(&self, path: &str) -> Option<&ChunkEntry> {
        self.directory.chunks.get(path)
    }
    
    /// Private methods
    fn write_header(&mut self) -> Result<(), MMXError> {
        let mut writer = BufWriter::new(&mut self.file);
        write_header(&mut writer, &self.header)
    }
    
    fn write_chunk(
        &mut self,
        path: &str,
        chunk_type: ChunkType,
        data: Vec<u8>,
        compression: CompressionType
    ) -> Result<(), MMXError> {
        let compressed_data = compress_data(&data, compression)?;
        let checksum = calculate_checksum(&compressed_data);
        
        // Write data to file (simplified)
        let offset = self.file.seek(SeekFrom::End(0))?;
        use std::io::Write;
        self.file.write_all(&compressed_data)?;
        
        // Update directory
        let entry = ChunkEntry {
            name: path.to_string(),
            chunk_type,
            offset,
            size: compressed_data.len() as u64,
            compression,
            shape: None,
            dtype: None,
            logical_region: None,
            checksum: Some(checksum),
        };
        
        self.directory.chunks.insert(path.to_string(), entry);
        self.header.total_chunks += 1;
        
        Ok(())
    }
    
    fn read_chunk(&mut self, path: &str) -> Result<Vec<u8>, MMXError> {
        let entry = self.directory.chunks.get(path)
            .ok_or_else(|| MMXError::ChunkNotFound(path.to_string()))?;
        
        // Seek to chunk position
        self.file.seek(SeekFrom::Start(entry.offset))?;
        
        // Read compressed data
        let mut compressed_data = vec![0u8; entry.size as usize];
        use std::io::Read;
        self.file.read_exact(&mut compressed_data)?;
        
        // Verify checksum if present
        if let Some(expected_checksum) = entry.checksum {
            let actual_checksum = calculate_checksum(&compressed_data);
            if actual_checksum != expected_checksum {
                return Err(MMXError::DataIntegrity("Checksum mismatch".to_string()));
            }
        }
        
        // Decompress
        decompress_data(&compressed_data, entry.compression)
    }
}

impl Drop for MMXFile {
    fn drop(&mut self) {
        // TODO: persist directory to file on close
    }
}

/// High-level convenience API
pub struct MMXBuilder {
    creator: String,
    model_uuid: Option<Uuid>,
}

impl MMXBuilder {
    pub fn new(creator: String) -> Self {
        Self {
            creator,
            model_uuid: None,
        }
    }
    
    pub fn with_model_uuid(mut self, uuid: Uuid) -> Self {
        self.model_uuid = Some(uuid);
        self
    }
    
    pub fn create<P: AsRef<Path>>(self, path: P) -> Result<MMXFile, MMXError> {
        MMXFile::create(path, self.creator)
    }
}
