import { useState } from "react";
import { useAuth } from "../auth";

export default function Login() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await login(username.trim(), password);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm bg-white border border-slate-200 rounded-xl shadow-sm p-6 space-y-4"
      >
        <h1 className="text-lg font-semibold text-center">כניסה לסוכן המחקר</h1>

        <div>
          <label className="block text-sm font-medium mb-1">שם משתמש</label>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
            dir="ltr"
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">סיסמה</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            dir="ltr"
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {err && (
          <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg p-2">
            {err}
          </div>
        )}

        <button
          type="submit"
          disabled={busy || !username || !password}
          className="w-full px-3 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {busy ? "מתחבר…" : "כניסה"}
        </button>

        <p className="text-xs text-slate-500 text-center pt-2">
          הסיסמה הראשונית של המנהל היא <code className="bg-slate-100 px-1 rounded">admin / admin</code>
        </p>
      </form>
    </div>
  );
}
