# PointnClick iPad/Tailscale Workflow

This workflow runs the model on the GPU PC and lets iPads control it from WebKnossos through a Safari extension.

## Architecture

- The GPU PC runs `webknossos-remote-server`.
- The server loads `best_model.pt` once and keeps it hot in GPU memory.
- Users create PointnClick accounts on the server dashboard.
- Each user saves their own WebKnossos auth token once.
- The iPad Safari extension injects a floating `Auto Mask` toggle into `webknossos.org`.
- When `Auto Mask` is on, a tap in WebKnossos sends the current position and active segment to the GPU PC. The server reads raw WebKnossos data, runs the model, and sends mask runs back to the extension for painting.

## 1. Set Up Tailscale

On the GPU PC:

1. Install Tailscale.
2. Sign in to your tailnet.
3. Find the PC's Tailscale IP:

```powershell
tailscale ip -4
```

It should look like `100.x.y.z`.

On each iPad:

1. Install the Tailscale app.
2. Sign in to the same tailnet.
3. Keep Tailscale connected while annotating.

Use the direct Tailscale IP for the bridge URL:

```text
http://100.x.y.z:8765
```

This is the simplest and lowest-overhead route for internal use.

## 2. Install Server Dependencies

On the GPU PC, from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_pointnclick_env.ps1
```

This installs CUDA PyTorch, WebKnossos, and `cryptography` into the `pointnclick` conda environment.

## 3. Start The GPU Server

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pointnclick_tailscale_server.ps1 -Checkpoint ".\runs\runs\worm_unet\best_model.pt"
```

If Windows asks about firewall access, allow access on private networks. The server listens on `0.0.0.0:8765`, which lets Tailscale reach it.

To confirm it is running from the GPU PC:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

To confirm it is reachable from an iPad, open this in Safari:

```text
http://100.x.y.z:8765/
```

## 4. Create User Accounts

On the iPad, open:

```text
http://100.x.y.z:8765/
```

Then:

1. Create a username/password.
2. Paste the user's WebKnossos auth token.
3. Save the token.

The token is encrypted at rest in the server database under:

```text
%LOCALAPPDATA%\PointnClick\remote_webknossos_server
```

## 5. Package The Safari Extension On A Mac

On the Mac, clone or copy this repo, then run:

```bash
xcrun safari-web-extension-converter examples/webknossos_chrome_extension \
  --project-location build/safari \
  --app-name PointnClickWebKnossos \
  --bundle-identifier org.pointnclick.webknossos
```

Open the generated Xcode project, select your Apple development team, and run the iOS/iPadOS app target on the iPad. For multiple users, distribute it through TestFlight or your normal internal Apple deployment route.

## 6. Configure The Extension On iPad

In Safari on the iPad:

1. Enable the PointnClick Safari extension.
2. Open the extension popup.
3. Set `Bridge URL` to:

```text
http://100.x.y.z:8765
```

4. Enter the PointnClick username/password.
5. Tap `Sign in`.
6. Tap `Test`.
7. Open or refresh `https://webknossos.org/.../view`.

Inside WebKnossos, the extension adds a floating `Auto Mask` button. Turn it on, select the segment/color in WebKnossos, and tap where you want a model mask. Turn it off to return to normal manual editing.

## Troubleshooting

- If `Test` fails, confirm the iPad and GPU PC are both connected to Tailscale.
- If the dashboard does not load from the iPad, confirm the server is running and Windows Firewall allowed Python.
- If the extension is signed in but prediction fails, open the dashboard and confirm the user's WebKnossos token is saved.
- If masks paint into the wrong segment, select the desired segment/color in WebKnossos before tapping.
- If Auto Mask fires while using WebKnossos controls, turn it off before changing tools or settings.
