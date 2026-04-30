# Round01 Stage A 报告：发送 Instagram 请求与邮件确认

报告时间：`2026-04-28`

最新更新：`2026-04-29`

沟通更新：`2026-04-29 17:08 Australia/Sydney`

后续沟通更新：`2026-04-29 17:14 Australia/Sydney`

进展同步更新：`2026-04-29 17:15 Australia/Sydney`

Access 更新：`2026-04-29 17:19 Australia/Sydney`

## 一、Stage 定位
- Stage 主题：添加 NLP 队友 Instagram 并告知已发送请求。
- 面向对象：协作者 / 接手人
- 核心目标：建立 COMP90042 A3 小组沟通渠道的第一步。

## 二、本轮背景与范围
- 当前项目阶段：团队协作初始化。
- 本轮要解决的问题：队友请求用户通过 Instagram 加入 group chat。
- 本轮包含的范围：Gmail 线程确认 handle、Instagram 发送请求、Gmail 回复。
- 本轮不包含的范围：反向搜索额外社交账号、课程项目模型实现、group chat 内后续任务分配。

## 三、实际进度
- 已完成：
  - 在 Gmail 线程 `Re: NLP Group assignment` 中确认 Kaitlyn 和 Om 主动提供了 Instagram handle。
  - 使用 Edge 登录后的 Instagram 会话打开两个账号主页。
  - 向 `@kaitlynkc36` 发送请求，页面状态为 `已发送`。
  - 向 `@x_omsubratodey_x` 发送请求，页面状态为 `已发送`。
  - 回复原 Gmail 线程，说明已经向两位发送 Instagram 请求。
- 已完成更新：
  - 2026-04-29 检查 Gmail 线程，最新邮件仍是用户 2026-04-28 发出的“已发送 Instagram 请求”，队友暂无邮件回复。
  - 2026-04-29 检查 Instagram 私信，`@kaitlynkc36` 和 `@x_omsubratodey_x` 均显示“你们成为好友啦。打个招呼吧！”。
  - Instagram 私信入口可用，说明两位已接受请求或已可直接沟通。
  - Om 的私信可见消息：`Hey` 和 `Send ur whatsapp`。
  - Kaitlyn 对话未见新的正文消息。
  - Instagram 陌生消息页未见额外可处理请求列表。
  - 已在 Instagram 创建三人群聊：`Om Subrato Dey 和 Kaitlyn`。
  - 已在群聊发送消息：`hey guys, i can't really use whatsapp atm - is it cool if we just chat here on ig for the project?`
  - 已说明 WhatsApp 暂时不可用，建议短聊用 IG，文件用 Google Drive/email。
  - 已提出可用 GitHub 管理 code/notebook，数据继续放 Drive/email。
  - Om 回复：`Yep that works as of now`
  - Om 表示：`Will start the work from tmr onwards strictly`
  - Om 表示 Kaitlyn 已经创建并完成了一些相关工作，让用户查看。
  - 已回复：`oh nice, can you send me the link/files? i'll have a look and build from there`
  - Om 发送了一个富媒体/附件消息，并 tag Kaitlyn：`@kaitlynkc36 give her access`
  - 已同步本地进展：`thanks. just to update, i also got the files/eval running locally and tried a basic tf-idf retrieval baseline. nothing final, just checking the pipeline. i'll compare with Kaitlyn's stuff once i get access`
  - Om 要求发送 personal email 用于 access。
  - Om 表示他们已经整理了一些数据；如果有 tentative code/idea，可以用合适名称放到 GDrive，但添加前先告知 Om 和 Kaitlyn。
  - 已发送工作邮箱：`jevorianx@gmail.com`
  - 已说明：拿到 Drive access 后会先查看，今天晚些时候再添加已尝试内容和 rough next-step plan，且不会先做大的改动。
- 未完成但已识别：
  - 尚未收到 Kaitlyn 对已有内容的 access。

## 四、核心交付
- 产出物：两个 Instagram 请求已发送；邮件回复已发送；本 Agent Docs round 记录已建立。
- 最重要结论：本轮主动操作部分已完成，剩余工作是等待对方接受。
- 对后续 stage 或下一轮的直接价值：通过后可以开始小组沟通、任务拆分和项目方案讨论。
- 当前阶段判断：团队 Instagram 群聊已建立，Om 接受暂时用 IG/Drive/email；已提供 `jevorianx@gmail.com`，当前等待 Kaitlyn 给 Drive access。

## 五、实现链路
1. 使用 Gmail MCP 搜索 NLP group assignment 相关邮件。
2. 读取原线程并确认明确提供的 handles：`@kaitlynkc36`、`@x_omsubratodey_x`。
3. 使用 Edge 远程调试会话进入 Instagram，用户手动完成登录。
4. 逐个打开账号主页并点击关注/请求按钮。
5. 读取页面按钮状态，确认两者均显示 `已发送`。
6. 回复 Gmail 原线程，告知已发送 Instagram 请求。
7. 创建 Round01 记录，保留为 `进行中`。

## 六、当前结果
- 已得到的结果：
  - `@kaitlynkc36`：Instagram 请求已发送。
  - `@x_omsubratodey_x`：Instagram 请求已发送。
  - Gmail 回复已发送到原线程。
- 已验证的内容：
  - 两个账号主页按钮状态均读取为 `已发送`。
  - Gmail 发送接口返回已发送邮件记录。
- 已验证的内容：
  - 两位队友均已和用户成为 Instagram 好友。
  - Instagram 私信可直接沟通。
  - 三人 Instagram group chat 已创建并已发送第一条沟通方式确认消息。
- 未验证的内容：
  - Kaitlyn 已创建/完成内容的具体位置和访问权限。
- 当前可以支撑哪些后续工作：
  - 等待通过后关闭 Round01。
  - 进入小组沟通后开启项目方案或任务分工 round。

## 七、风险、不足与阻塞
- 当前不足：还未拿到 Kaitlyn 已做内容的 access。
- 风险点：如果本地 Round03/Round04 工作和 Kaitlyn 的工作重复，需要先对齐再合并。
- 阻塞项：等待 Kaitlyn 对 `jevorianx@gmail.com` 授权或发送 Drive 链接/文件。
- 对后续推进的影响：技术工作可继续推进；任务分工最好尽快通过 Instagram 或 WhatsApp 确认。

## 八、下一步建议
- 优先级 1：等待 Kaitlyn 对 `jevorianx@gmail.com` 授权或发送已有工作链接/文件。
- 优先级 2：审计 Drive 已有工作和本地 Round03/Round04 的重叠，再决定 GitHub/Drive 组织方式。
- 优先级 3：技术侧继续 Round04 retrieval baseline。

## 九、管理结论
- 一句话总结当前 stage 状态：两位队友已成为 Instagram 好友，三人 IG 群聊已创建，Om 接受暂时用 IG/Drive/email，并提示 Kaitlyn 已有部分工作；Round01 可视为达标。
- 是否建议关闭当前 stage：建议关闭。
- 继续推进的最关键前提：确定是否使用 WhatsApp/group chat 做后续小组沟通。
