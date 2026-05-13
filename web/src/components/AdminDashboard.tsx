import { useEffect, useState } from "react";
import {
  createManagedUser,
  disableUser,
  downloadExport,
  listManagedUsers,
  listUserSessions,
  updateUserBudget,
} from "../api";
import type { ManagedUser, SessionOut } from "../types";

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
    </div>
  );
}
