# Round01 索引：添加 NLP 队友 Instagram 并进入沟通渠道

报告时间：`2026-04-28`

## 一、本轮总目标
- 使用队友在 Gmail 中明确提供的 Instagram handle，向 NLP team members 发送连接请求。
- 回复邮件告知已经发送请求。
- 等待两位队友通过请求后，将本轮标记为完成。

## 二、全局约束
- 仅使用邮件中主动提供的 handle：`@kaitlynkc36` 和 `@x_omsubratodey_x`。
- 不基于姓名、邮箱、学校身份等做额外社交账号反查。
- 网站登录由用户在 Edge 中手动完成；Codex 登录后继续自动化。

## 三、Stage 划分
### Stage A：发送 Instagram 请求与邮件确认
报告：

```txt
agent_docs/rounds/round_01/round_01_stage_a_instagram_requests_report.md
```

状态：`已达标`

## 四、跨 Stage 依赖
- 只有 Stage A；本 round 的剩余依赖是外部等待：队友通过 Instagram 请求。

## 五、关闭条件
- `@kaitlynkc36` 请求状态不再是 `已发送`，并确认已通过或可以沟通。
- `@x_omsubratodey_x` 请求状态不再是 `已发送`，并确认已通过或可以沟通。
- 若两位队友将用户加入 group chat，也可视为本 round 完成。

## 六、下一步
- 技术侧继续 Round04 retrieval baseline。
- 沟通侧需要用户决定是否向 Om 回复 WhatsApp 联系方式，或改为请队友继续使用 Instagram/group chat。
