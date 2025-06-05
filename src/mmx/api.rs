//! High-level API for working with MMX files

use super::{
    MMXError, MMXHeader, ChunkDirectory, ChunkEntry, ChunkType, CompressionType,
    TensorData, SequenceData, TextData, MeshData, EmbeddingData, MetaData, GeometricTemplateData,
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
            .read(true)
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
        
        // Attempt to read directory from end of file. If this fails, fall back
        // to an empty directory so legacy files still open.
        let directory = match read_directory(&mut reader) {
            Ok(dir) => dir,
            Err(_) => ChunkDirectory { chunks: HashMap::new() },
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

    /// Write sequence chunk
    pub fn write_sequence(&mut self, path: &str, seq: SequenceData) -> Result<(), MMXError> {
        let data = bincode::serialize(&seq).map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        self.write_chunk(path, ChunkType::Sequence, data, CompressionType::Lz4)
    }

    /// Read sequence chunk
    pub fn read_sequence(&mut self, path: &str) -> Result<SequenceData, MMXError> {
        let data = self.read_chunk(path)?;
        bincode::deserialize(&data).map_err(|e| MMXError::DataIntegrity(e.to_string()))
    }

    /// Write text chunk
    pub fn write_text(&mut self, path: &str, text: TextData) -> Result<(), MMXError> {
        let data = bincode::serialize(&text).map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        self.write_chunk(path, ChunkType::Text, data, CompressionType::Lz4)
    }

    /// Read text chunk
    pub fn read_text(&mut self, path: &str) -> Result<TextData, MMXError> {
        let data = self.read_chunk(path)?;
        bincode::deserialize(&data).map_err(|e| MMXError::DataIntegrity(e.to_string()))
    }

    /// Write metadata chunk
    pub fn write_metadata(&mut self, path: &str, metadata: MetaData) -> Result<(), MMXError> {
        let data = bincode::serialize(&metadata).map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        self.write_chunk(path, ChunkType::Meta, data, CompressionType::Lz4)
    }

    /// Read metadata chunk
    pub fn read_metadata(&mut self, path: &str) -> Result<MetaData, MMXError> {
        let data = self.read_chunk(path)?;
        bincode::deserialize(&data).map_err(|e| MMXError::DataIntegrity(e.to_string()))
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

    /// Write geometric template chunk
    pub fn write_geometric_template(&mut self, path: &str, template: GeometricTemplateData) -> Result<(), MMXError> {
        let data = template.to_bytes();
        self.write_chunk(path, ChunkType::GeometricTemplate, data, CompressionType::Lz4)
    }

    /// Read geometric template chunk
    pub fn read_geometric_template(&mut self, path: &str) -> Result<GeometricTemplateData, MMXError> {
        let data = self.read_chunk(path)?;
        GeometricTemplateData::from_bytes(&data).map_err(|e| MMXError::DataIntegrity(e.to_string()))
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
        // Only persist on writable modes. Errors are ignored in Drop
        if matches!(self.mode, MMXMode::Write | MMXMode::Append) {
            if let Ok(mut writer) = self.file.try_clone() {
                if write_directory(&mut writer, &self.directory).is_ok() {
                    if let Ok(size) = writer.seek(SeekFrom::End(0)) {
                        self.header.file_size = size;
                        let _ = writer.seek(SeekFrom::Start(0));
                        let _ = write_header(&mut writer, &self.header);
                    }
                }
            }
        }
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
