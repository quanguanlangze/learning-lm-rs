use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

//- safetensors里存储的是原始数据，你需要以FP32的形式读取出来，创建出项目所使用的张量。
// - safetensors包含张量的形状，你无需对原始张量做任何变形。
// - 当"tie_word_embeddings"属性被打开时，模型最开始以及最后的embedding矩阵数据相同，safetensors会只存储一份数据，我们测试用的story模型就是这样。作业阶段你可以只关心story模型，但是后续项目中你需要处理两个矩阵不同的情况。
impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let layers = config.num_hidden_layers;

        fn get_tensor(safetensor: &SafeTensors, name: &str) -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).unwrap();
            let data: Vec<f32> = tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().expect("slice with incorrect length");
                    f32::from_le_bytes(bytes)
                })
                .collect();
            Tensor::new(data, &tensor_view.shape().to_vec())
        }

        fn get_layer_tensors(
            safetensor: &SafeTensors,
            prefix: &str,
            layers: usize,
            suffix: &str,
        ) -> Vec<Tensor<f32>> {
            (0..layers)
                .map(|i| get_tensor(safetensor, &format!("model.layers.{i}.{}.{}", prefix, suffix)))
                .collect()
        }

        Self {
            embedding_table: get_tensor(safetensor, "lm_head.weight"),
            rms_att_w: get_layer_tensors(safetensor, "input_layernorm", layers, "weight"),
            wq: get_layer_tensors(safetensor, "self_attn.q_proj", layers, "weight"),
            wk: get_layer_tensors(safetensor, "self_attn.k_proj", layers, "weight"),
            wv: get_layer_tensors(safetensor, "self_attn.v_proj", layers, "weight"),
            wo: get_layer_tensors(safetensor, "self_attn.o_proj", layers, "weight"),
            rms_ffn_w: get_layer_tensors(safetensor, "post_attention_layernorm", layers, "weight"),
            w_up: get_layer_tensors(safetensor, "mlp.up_proj", layers, "weight"),
            w_gate: get_layer_tensors(safetensor, "mlp.gate_proj", layers, "weight"),
            w_down: get_layer_tensors(safetensor, "mlp.down_proj", layers, "weight"),
            rms_out_w: get_tensor(safetensor, "model.norm.weight"),
            lm_head: get_tensor(safetensor, "lm_head.weight"),
        }
    }
}
