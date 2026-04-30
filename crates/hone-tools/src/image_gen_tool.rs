//! ImageGenTool — 图像生成工具
//!
//! 包装 hone_integrations::NanoBananaClient, 通过 OpenRouter 的图像生成
//! 模型 (默认 google/gemini-3-pro-image-preview) 出图, 然后把图片落到
//! 当前 actor 的 gen_images 目录下.
//!
//! 与 image_generation skill 的 SKILL.md 协议对齐: 接收 image_type
//! (general / portfolio_snapshot / stock_analysis), prompt 与 content 二选一
//! 或同时给, 内部拼成一段送给 NanoBanana 的文本.
//!
//! 之前 image_generation skill 引用 image_gen 这个 tool, 但 honeclaw 没有
//! 在 ToolRegistry 注册任何 image_gen 实现, 导致 LLM 拿到 SKILL.md 后
//! 只能假装"已生成", 实际没有图片落地. 该工具修复这个静默失败.

use async_trait::async_trait;
use hone_core::{ActorIdentity, HoneError, HoneResult};
use hone_integrations::NanoBananaClient;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::base::{Tool, ToolParameter};

const DEFAULT_DRAFT_PREFIX: &str = "image_gen";
const ALLOWED_IMAGE_TYPES: &[&str] = &["general", "portfolio_snapshot", "stock_analysis"];
const MAX_PROMPT_CHARS: usize = 4000;

pub struct ImageGenTool {
    client: Arc<NanoBananaClient>,
    actor: ActorIdentity,
}

impl ImageGenTool {
    pub fn new(client: Arc<NanoBananaClient>, actor: ActorIdentity) -> Self {
        Self { client, actor }
    }

    pub fn from_config(config: &hone_core::config::HoneConfig, actor: ActorIdentity) -> Self {
        Self {
            client: Arc::new(NanoBananaClient::from_config(config)),
            actor,
        }
    }

    fn build_prompt(image_type: &str, prompt: Option<&str>, content: Option<&str>) -> String {
        // 三类的 prompt 合成策略不同:
        //   general          → 直接用 prompt
        //   portfolio_snapshot → 模板 + content (持仓清单)
        //   stock_analysis    → 模板 + content (公司分析数据)
        // 这里复用 SKILL.md 描述的语义, 提示模型生成研究风格图.
        let user_prompt = prompt.unwrap_or("").trim();
        let user_content = content.unwrap_or("").trim();

        match image_type {
            "portfolio_snapshot" => format!(
                "Generate a clean, professional portfolio dashboard infographic.\n\
                 Style: financial research, minimalist, dark blue accents, sans-serif.\n\
                 Render the holdings table clearly with ticker, name, price and percent change.\n\n\
                 Holdings:\n{user_content}\n\n\
                 Extra style notes: {user_prompt}"
            ),
            "stock_analysis" => format!(
                "Generate a single-stock analysis infographic in research-report style.\n\
                 Style: clean, data-dense, professional financial report aesthetic.\n\
                 Highlight the key numbers and the moat / thesis sentences.\n\n\
                 Stock data and thesis:\n{user_content}\n\n\
                 Extra style notes: {user_prompt}"
            ),
            _ => {
                // general
                if user_content.is_empty() {
                    user_prompt.to_string()
                } else {
                    format!("{user_prompt}\n\n{user_content}").trim().to_string()
                }
            }
        }
    }
}

#[async_trait]
impl Tool for ImageGenTool {
    fn name(&self) -> &str {
        "image_gen"
    }

    fn description(&self) -> &str {
        "调用 OpenRouter 图像生成模型 (NanoBanana / gemini image preview) 出图, \
         自动把生成的图片落到当前 actor 的 gen_images 目录, 返回本地绝对路径.\n\n\
         三种 image_type:\n\
         - general: 直接按 prompt 生成图片\n\
         - portfolio_snapshot: 把组合数据 (content) 渲染成 dashboard 信息图\n\
         - stock_analysis: 把单股数据 (content) 渲染成研究报告风格信息图\n\n\
         portfolio_snapshot 与 stock_analysis 必须先调用 portfolio / data_fetch 等工具拿真实数字, \
         再把整理好的文本作为 content 传入, 不要让模型自己编数据."
    }

    fn parameters(&self) -> Vec<ToolParameter> {
        vec![
            ToolParameter {
                name: "image_type".to_string(),
                param_type: "string".to_string(),
                description: "图片类型: general / portfolio_snapshot / stock_analysis."
                    .to_string(),
                required: true,
                r#enum: Some(ALLOWED_IMAGE_TYPES.iter().map(|s| s.to_string()).collect()),
                items: None,
            },
            ToolParameter {
                name: "prompt".to_string(),
                param_type: "string".to_string(),
                description: "image_type=general 时是主文案; 其它类型是补充风格描述, 可选."
                    .to_string(),
                required: false,
                r#enum: None,
                items: None,
            },
            ToolParameter {
                name: "content".to_string(),
                param_type: "string".to_string(),
                description: "结构化数据文本: portfolio_snapshot 用持仓清单, stock_analysis 用单股关键数字与 thesis. general 时可选."
                    .to_string(),
                required: false,
                r#enum: None,
                items: None,
            },
            ToolParameter {
                name: "count".to_string(),
                param_type: "number".to_string(),
                description: "返回的图片数, 默认走 config.nano_banana.default_image_count.".to_string(),
                required: false,
                r#enum: None,
                items: None,
            },
        ]
    }

    async fn execute(&self, args: Value) -> HoneResult<Value> {
        let image_type = args
            .get("image_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_lowercase();
        if !ALLOWED_IMAGE_TYPES.contains(&image_type.as_str()) {
            return Err(HoneError::Tool(format!(
                "image_type 不支持: {image_type} (允许 {})",
                ALLOWED_IMAGE_TYPES.join(", ")
            )));
        }

        let prompt = args.get("prompt").and_then(|v| v.as_str());
        let content = args.get("content").and_then(|v| v.as_str());
        if prompt.unwrap_or("").trim().is_empty() && content.unwrap_or("").trim().is_empty() {
            return Err(HoneError::Tool(
                "prompt 与 content 至少需要提供一个非空字符串".to_string(),
            ));
        }

        let composed = Self::build_prompt(&image_type, prompt, content);
        if composed.chars().count() > MAX_PROMPT_CHARS {
            return Err(HoneError::Tool(format!(
                "合成 prompt 长度超过 {MAX_PROMPT_CHARS} 字符上限"
            )));
        }

        let count = args
            .get("count")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let gen_result = self.client.generate_images(&composed, count).await;
        let success = gen_result
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !success {
            let err_msg = gen_result
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("OpenRouter 出图调用失败")
                .to_string();
            return Err(HoneError::Tool(err_msg));
        }

        let urls: Vec<String> = gen_result
            .get("image_urls")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if urls.is_empty() {
            return Err(HoneError::Tool(
                "OpenRouter 返回成功但未提取到任何图片 URL".to_string(),
            ));
        }

        let draft_id = format!(
            "{DEFAULT_DRAFT_PREFIX}_{}_{}",
            image_type,
            chrono::Utc::now().format("%Y%m%d%H%M%S")
        );
        let local_paths = self
            .client
            .download_images(&urls, &self.actor, &draft_id)
            .await?;

        Ok(json!({
            "image_type": image_type,
            "image_urls": urls,
            "local_paths": local_paths,
            "task_id": gen_result.get("task_id").and_then(|v| v.as_str()).unwrap_or(""),
            "count": local_paths.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_general_uses_user_text() {
        let p = ImageGenTool::build_prompt("general", Some("a red square"), None);
        assert!(p.contains("red square"));
    }

    #[test]
    fn build_prompt_portfolio_includes_holdings_and_style() {
        let p = ImageGenTool::build_prompt(
            "portfolio_snapshot",
            Some("dark theme"),
            Some("AAPL Apple $189.50 +1.2%"),
        );
        assert!(p.contains("AAPL Apple"));
        assert!(p.contains("dark theme"));
        assert!(p.contains("portfolio dashboard"));
    }

    #[test]
    fn build_prompt_stock_analysis_includes_thesis() {
        let p = ImageGenTool::build_prompt(
            "stock_analysis",
            None,
            Some("NVDA Market cap 3.2T Moat: CUDA"),
        );
        assert!(p.contains("CUDA"));
        assert!(p.contains("research-report"));
    }
}
