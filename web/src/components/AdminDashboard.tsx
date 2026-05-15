import { useEffect, useState } from "react";
import {
  createManagedUser,
  disableUser,
  downloadExport,
  getMemoryStats,
  listManagedUsers,
  listUserSessions,
  seedMemory,
  updateUserBudget,
} from "../api";
import type { ManagedUser, MemoryStats, SessionOut } from "../types";

export default function AdminDashboard() {
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newBudget, setNewBudget] = useState("0");
  const [creating, setCreating] = useState(false);

  const [selectedUser, setSelectedUser] = useState<ManagedUser | null>(null);
  const [userSessions, setUserSessions] = useState<SessionOut[]>([]);

  async function refresh() {
    setLoading(true);
    setErr(null);
    try {
      setUsers(await listManagedUsers());
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setErr(null);
    try {
      await createManagedUser({
        username: newUsername.trim(),
        password: newPassword,
        credit_balance: parseFloat(newBudget) || 0,
      });
      setNewUsername("");
      setNewPassword("");
      setNewBudget("0");
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function onAddBudget(user: ManagedUser) {
    const raw = prompt(`כמה לסכום הקיים של ${user.username}? (USD)`, "1.00");
    if (raw === null) return;
    const amount = parseFloat(raw);
    if (!isFinite(amount)) return;
    try {
      await updateUserBudget(user.id, { add: amount });
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function onSetBudget(user: ManagedUser) {
    const raw = prompt(`קבע יתרה חדשה עבור ${user.username} (USD)`, user.credit_balance.toFixed(2));
    if (raw === null) return;
    const amount = parseFloat(raw);
    if (!isFinite(amount)) return;
    try {
      await updateUserBudget(user.id, { set_to: amount });
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function onDisable(user: ManagedUser) {
    if (!confirm(`לבטל את חשבון ${user.username}?`)) return;
    try {
      await disableUser(user.id);
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function onSelect(user: ManagedUser) {
    setSelectedUser(user);
    setUserSessions([]);
    try {
      const sessions = await listUserSessions(user.id);
      setUserSessions(sessions);
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  return (
    <div className="space-y-4">
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        <h2 className="text-base font-semibold mb-3">צור משתמש חדש</h2>
        <form onSubmit={onCreate} className="grid grid-cols-1 md:grid-cols-4 gap-2 items-end">
          <div>
            <label className="block text-xs text-slate-600 mb-1">שם משתמש</label>
            <input
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              dir="ltr"
              required
              className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-600 mb-1">סיסמה</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              dir="ltr"
              required
              className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-600 mb-1">תקציב ראשוני (USD)</label>
            <input
              type="number"
              step="0.01"
              value={newBudget}
              onChange={(e) => setNewBudget(e.target.value)}
              dir="ltr"
              className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
            />
          </div>
          <button
            type="submit"
            disabled={creating}
            className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {creating ? "יוצר…" : "צור משתמש"}
          </button>
        </form>
      </div>

      {err && (
        <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg p-3">
          {err}
        </div>
      )}

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-x-auto">
        <table className="text-sm w-full">
          <thead className="bg-slate-100 text-right">
            <tr>
              <th className="p-2">משתמש</th>
              <th className="p-2">סטטוס</th>
              <th className="p-2">יתרה (USD)</th>
              <th className="p-2">סך עלות שנצברה</th>
              <th className="p-2">מחקרים</th>
              <th className="p-2">פעולות</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6} className="p-4 text-center text-slate-500">טוען…</td>
              </tr>
            )}
            {!loading && users.length === 0 && (
              <tr>
                <td colSpan={6} className="p-4 text-center text-slate-500">אין משתמשים תחתיך עדיין</td>
              </tr>
            )}
            {users.map((u) => (
              <tr key={u.id} className="border-t border-slate-100">
                <td className="p-2 font-medium" dir="ltr">{u.username}</td>
                <td className="p-2">
                  {u.is_active ? (
                    <span className="text-emerald-700">פעיל</span>
                  ) : (
                    <span className="text-slate-400">מבוטל</span>
                  )}
                </td>
                <td className="p-2 font-mono">{u.credit_balance.toFixed(2)}</td>
                <td className="p-2 font-mono">{u.total_cost_used.toFixed(4)}</td>
                <td className="p-2 font-mono">{u.session_count}</td>
                <td className="p-2 space-x-2 space-x-reverse">
                  <button
                    onClick={() => onAddBudget(u)}
                    className="text-blue-600 hover:underline text-xs"
                  >
                    הוסף כסף
                  </button>
                  <button
                    onClick={() => onSetBudget(u)}
                    className="text-blue-600 hover:underline text-xs"
                  >
                    קבע יתרה
                  </button>
                  <button
                    onClick={() => onSelect(u)}
                    className="text-blue-600 hover:underline text-xs"
                  >
                    היסטוריה
                  </button>
                  {u.is_active && (
                    <button
                      onClick={() => onDisable(u)}
                      className="text-rose-600 hover:underline text-xs"
                    >
                      בטל
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedUser && (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-x-auto">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between">
            <h3 className="font-semibold">
              היסטוריית מחקרים של <span dir="ltr">{selectedUser.username}</span>
            </h3>
            <button
              onClick={() => setSelectedUser(null)}
              className="text-xs text-slate-500 hover:text-slate-700"
            >
              סגור
            </button>
          </div>
          <table className="text-sm w-full">
            <thead className="bg-slate-50 text-right">
              <tr>
                <th className="p-2 w-1/2">שאלה</th>
                <th className="p-2">סטטוס</th>
                <th className="p-2">עלות</th>
                <th className="p-2">תאריך</th>
                <th className="p-2">קובץ</th>
              </tr>
            </thead>
            <tbody>
              {userSessions.length === 0 && (
                <tr>
                  <td colSpan={5} className="p-4 text-center text-slate-500">
                    אין מחקרים עדיין
                  </td>
                </tr>
              )}
              {userSessions.map((s) => (
                <tr key={s.id} className="border-t border-slate-100 align-top">
                  <td className="p-2" dir="auto">{s.question}</td>
                  <td className="p-2">{s.status}</td>
                  <td className="p-2 font-mono">{s.cost_used.toFixed(4)}</td>
                  <td className="p-2 text-slate-600">
                    {new Date(s.created_at).toLocaleString("he-IL")}
                  </td>
                  <td className="p-2">
                    {s.exports.map((x) => (
                      <button
                        key={x.id}
                        onClick={() => downloadExport(x.id, x.filename)}
                        className="text-blue-600 hover:underline text-xs block"
                      >
                        ⬇ {x.filename}
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <MemorySection />
    </div>
  );
}


// ── Memory stats + manual seeding ────────────────────────────────────────────

function MemorySection() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  // Seed form state
  const [kind, setKind] = useState<"success" | "failure">("success");
  const [fieldType, setFieldType] = useState("person_name");
  const [fieldLabel, setFieldLabel] = useState("");
  const [entity, setEntity] = useState("");
  const [value, setValue] = useState("");
  const [quote, setQuote] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [reason, setReason] = useState("");
  const [seeding, setSeeding] = useState(false);
  const [seedMsg, setSeedMsg] = useState<string | null>(null);
  const [seedErr, setSeedErr] = useState<string | null>(null);

  useEffect(() => {
    getMemoryStats().then(setStats).catch((e) => setLoadErr(e.message));
  }, []);

  async function handleSeed() {
    if (!fieldLabel || !entity || !value || !quote || !sourceUrl) return;
    if (kind === "failure" && !reason) return;
    setSeeding(true);
    setSeedMsg(null);
    setSeedErr(null);
    try {
      const res = await seedMemory({ kind, field_type: fieldType, field_label: fieldLabel, entity, value, quote, source_url: sourceUrl, reason: reason || null });
      setStats(res.stats);
      setSeedMsg(`נשמר (id: ${res.id.slice(0, 8)}…)`);
      setFieldLabel(""); setEntity(""); setValue(""); setQuote(""); setSourceUrl(""); setReason("");
    } catch (e) {
      setSeedErr((e as Error).message);
    } finally {
      setSeeding(false);
    }
  }

  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-4">
      <h2 className="font-semibold text-base">זיכרון המערכת</h2>
      {loadErr && <div className="text-xs text-rose-700">{loadErr}</div>}
      {stats && (
        <div className="flex gap-6 text-sm">
          <div><span className="font-mono text-blue-700">{stats.compiler_successes}</span> תוכניות מאומתות</div>
          <div><span className="font-mono text-emerald-700">{stats.extraction_successes}</span> חילוצים נכונים</div>
          <div><span className="font-mono text-rose-700">{stats.extraction_failures}</span> כשלונות מסומנים</div>
          <div className="text-slate-400 text-xs self-center">{stats.path}</div>
        </div>
      )}

      <details>
        <summary className="cursor-pointer text-sm font-medium text-slate-700 select-none">הוסף דוגמה ידנית</summary>
        <div className="mt-3 space-y-3 text-sm">
          <div className="flex gap-3 flex-wrap">
            <div>
              <label className="block text-xs text-slate-500 mb-1">סוג</label>
              <select value={kind} onChange={(e) => setKind(e.target.value as "success" | "failure")}
                className="border border-slate-300 rounded px-2 py-1 text-sm bg-white">
                <option value="success">נכון (דוגמה חיובית)</option>
                <option value="failure">שגוי (אזהרה)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-500 mb-1">סוג שדה</label>
              <select value={fieldType} onChange={(e) => setFieldType(e.target.value)}
                className="border border-slate-300 rounded px-2 py-1 text-sm bg-white">
                {["person_name","url","date","number","organization","free_text"].map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div className="flex-1 min-w-32">
              <label className="block text-xs text-slate-500 mb-1">שם שדה</label>
              <input value={fieldLabel} onChange={(e) => setFieldLabel(e.target.value)}
                placeholder="Mayor / ראש העיר" dir="auto"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
            <div className="flex-1 min-w-32">
              <label className="block text-xs text-slate-500 mb-1">ישות</label>
              <input value={entity} onChange={(e) => setEntity(e.target.value)}
                placeholder="Tel Aviv" dir="auto"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
          </div>
          <div className="flex gap-3 flex-wrap">
            <div className="flex-1 min-w-40">
              <label className="block text-xs text-slate-500 mb-1">ערך</label>
              <input value={value} onChange={(e) => setValue(e.target.value)}
                placeholder="Ron Huldai" dir="auto"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
            <div className="flex-1 min-w-64">
              <label className="block text-xs text-slate-500 mb-1">ציטוט מקורי (העתק-הדבק מהמקור)</label>
              <input value={quote} onChange={(e) => setQuote(e.target.value)}
                placeholder="Ron Huldai has served as mayor since 1998" dir="auto"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
            <div className="flex-1 min-w-48">
              <label className="block text-xs text-slate-500 mb-1">כתובת מקור</label>
              <input value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)}
                placeholder="https://..." dir="ltr"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
          </div>
          {kind === "failure" && (
            <div>
              <label className="block text-xs text-slate-500 mb-1">סיבת הכשל (חובה)</label>
              <input value={reason} onChange={(e) => setReason(e.target.value)}
                placeholder="הערך אינו מוזכר בטקסט — הלוצינציה" dir="auto"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm" />
            </div>
          )}
          {seedErr && <div className="text-xs text-rose-700">{seedErr}</div>}
          {seedMsg && <div className="text-xs text-emerald-700">{seedMsg}</div>}
          <div className="flex justify-end">
            <button onClick={handleSeed} disabled={seeding}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1.5 rounded-lg text-sm font-semibold disabled:opacity-50">
              {seeding ? "שומר…" : "שמור דוגמה"}
            </button>
          </div>
        </div>
      </details>
    </div>
  );
}
