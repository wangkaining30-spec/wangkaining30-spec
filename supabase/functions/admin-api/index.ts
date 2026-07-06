// Supabase Edge Function: admin-api
// Proxies admin operations, keeping service_role key server-side
// Deploy: supabase functions deploy admin-api --project-ref fegouedpioqqzsuazbeb

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = "https://fegouedpioqqzsuazbeb.supabase.co";
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OWNER_EMAIL = "kechen20120426@qq.com";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const url = new URL(req.url);
    const action = url.searchParams.get("action");
    const userId = url.searchParams.get("userId");

    // Auth: verify caller is owner/admin
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }

    const token = authHeader.replace("Bearer ", "");
    const adminClient = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
      auth: { persistSession: false }
    });

    // Get caller user
    const { data: { user: caller }, error: authErr } = await adminClient.auth.admin.getUserById(
      token.split(".")[1] ? (JSON.parse(atob(token.split(".")[1]))).sub : ""
    );

    // Fallback: verify via getUser
    const userClient = createClient(SUPABASE_URL, Deno.env.get("SUPABASE_ANON_KEY") || "", {
      auth: { persistSession: false }
    });
    const { data: { user: verifiedCaller }, error: verifyErr } = await userClient.auth.getUser(token);

    const callerUser = verifiedCaller || caller;
    if (!callerUser) {
      return new Response(JSON.stringify({ error: "Invalid token" }), {
        status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }

    // Check if caller is owner or admin
    const isOwner = callerUser.email === OWNER_EMAIL;
    const isAdmin = callerUser.app_metadata?.admin === true;
    if (!isOwner && !isAdmin) {
      return new Response(JSON.stringify({ error: "Forbidden: owner/admin only" }), {
        status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }

    let result;

    switch (action) {
      case "list_users": {
        const perPage = parseInt(url.searchParams.get("per_page") || "50");
        const { data, error } = await adminClient.auth.admin.listUsers({ perPage });
        if (error) throw error;
        result = data;
        break;
      }

      case "get_user": {
        if (!userId) throw new Error("userId required");
        const { data, error } = await adminClient.auth.admin.getUserById(userId);
        if (error) throw error;
        result = data;
        break;
      }

      case "update_user": {
        if (!userId) throw new Error("userId required");
        const body = await req.json();
        const { data, error } = await adminClient.auth.admin.updateUserById(userId, body);
        if (error) throw error;
        result = data;
        break;
      }

      case "delete_user": {
        if (!userId) throw new Error("userId required");
        const { data, error } = await adminClient.auth.admin.deleteUser(userId);
        if (error) throw error;
        result = { success: true };
        break;
      }

      default:
        return new Response(JSON.stringify({ error: `Unknown action: ${action}` }), {
          status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
    }

    return new Response(JSON.stringify(result), {
      status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });

  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
});
